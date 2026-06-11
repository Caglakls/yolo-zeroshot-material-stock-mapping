import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

from data_loader import BuildingRecord, ground_truth_materials, normalize_material
from materials_utils import compute_ratios, dominant_material

logger = logging.getLogger(__name__)


@dataclass
class MaterialMetrics:
    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    total: int
    correct: int


@dataclass
class WWRStats:
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    count: int


def evaluate_materials(
    predictions: dict[str, str],
    records: dict[str, BuildingRecord],
    alias_map: dict[str, str],
    material_labels: list[str],
) -> MaterialMetrics:
    """Compute material classification metrics.
    A prediction is correct if normalized value matches Front_wall or Sec_front_wall."""
    correct = 0
    total = 0

    # For per-class metrics: track true positives, false positives, false negatives
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = {m: {n: 0 for n in material_labels} for m in material_labels}

    for image_id, pred_raw in predictions.items():
        if image_id not in records:
            continue
        record = records[image_id]
        gt_materials = ground_truth_materials(record)
        if not gt_materials:
            continue

        pred_norm = normalize_material(pred_raw, alias_map)
        gt_primary = gt_materials[0]
        total += 1

        is_match = pred_norm in gt_materials
        if is_match:
            correct += 1
            tp[pred_norm] += 1
        else:
            fp[pred_norm] += 1
            fn[gt_primary] += 1

        # Confusion matrix: row = ground truth primary, col = prediction
        if gt_primary in confusion and pred_norm in confusion.get(gt_primary, {}):
            confusion[gt_primary][pred_norm] += 1

    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision, recall, F1
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    class_counts = defaultdict(int)

    for image_id in predictions:
        if image_id in records:
            gt = ground_truth_materials(records[image_id])
            if gt:
                class_counts[gt[0]] += 1

    for label in material_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_precision[label] = p
        per_class_recall[label] = r
        per_class_f1[label] = f1

    # Macro averages
    active_labels = [l for l in material_labels if (tp[l] + fp[l] + fn[l]) > 0]
    macro_p = sum(per_class_precision[l] for l in active_labels) / len(active_labels) if active_labels else 0.0
    macro_r = sum(per_class_recall[l] for l in active_labels) / len(active_labels) if active_labels else 0.0
    macro_f1 = sum(per_class_f1[l] for l in active_labels) / len(active_labels) if active_labels else 0.0

    # Weighted F1
    weighted_f1 = 0.0
    if total > 0:
        for l in material_labels:
            weighted_f1 += per_class_f1[l] * class_counts[l]
        weighted_f1 /= total

    return MaterialMetrics(
        accuracy=accuracy,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=confusion,
        total=total,
        correct=correct,
    )


def compute_wwr_stats(wwr_predictions: dict[str, float]) -> WWRStats:
    """Compute descriptive statistics for WWR predictions."""
    valid = [v for v in wwr_predictions.values() if 0.0 <= v <= 1.0]
    if not valid:
        return WWRStats(mean=0, median=0, std=0, min_val=0, max_val=0, count=0)

    valid.sort()
    n = len(valid)
    mean = sum(valid) / n
    median = valid[n // 2] if n % 2 == 1 else (valid[n // 2 - 1] + valid[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in valid) / n
    std = variance ** 0.5

    return WWRStats(
        mean=mean,
        median=median,
        std=std,
        min_val=valid[0],
        max_val=valid[-1],
        count=n,
    )


def save_detailed_results(
    output_path: str,
    image_records: list[BuildingRecord],
    claude_results: dict | None,
    clip_results: dict | None,
    llava_results: dict | None,
    alias_map: dict[str, str],
    wwr_model_results: dict[str, dict] | None = None,
):
    """Save per-image CSV with predictions vs ground truth."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "Image_ID",
        "Address",
        "GT_Front_Wall",
        "GT_Sec_Front_Wall",
    ]
    if claude_results is not None:
        fieldnames += ["Claude_Material", "Claude_Material_Norm", "Claude_WWR", "Claude_Match"]
    if clip_results is not None:
        fieldnames += ["CLIP_Material", "CLIP_Material_Norm", "CLIP_Match"]
    if llava_results is not None:
        fieldnames += ["LLaVA_Material", "LLaVA_Material_Norm", "LLaVA_WWR", "LLaVA_Match"]

    # Add columns for each WWR detection model
    wwr_model_names = []
    if wwr_model_results:
        for name in wwr_model_results:
            wwr_model_names.append(name)
            fieldnames += [f"{name}_WWR", f"{name}_Detections", f"{name}_Method"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in image_records:
            gt_mats = ground_truth_materials(record)
            row = {
                "Image_ID": record.image_id,
                "Address": record.address,
                "GT_Front_Wall": record.front_wall,
                "GT_Sec_Front_Wall": record.sec_front_wall,
            }

            if claude_results is not None and record.image_id in claude_results:
                cr = claude_results[record.image_id]
                norm = normalize_material(cr.material, alias_map)
                row["Claude_Material"] = cr.material
                row["Claude_Material_Norm"] = norm
                row["Claude_WWR"] = f"{cr.wwr:.3f}" if cr.wwr >= 0 else "N/A"
                row["Claude_Match"] = "Yes" if norm in gt_mats else "No"

            if clip_results is not None and record.image_id in clip_results:
                cr = clip_results[record.image_id]
                norm = normalize_material(cr.material, alias_map)
                row["CLIP_Material"] = cr.material
                row["CLIP_Material_Norm"] = norm
                row["CLIP_Match"] = "Yes" if norm in gt_mats else "No"

            if llava_results is not None and record.image_id in llava_results:
                lr = llava_results[record.image_id]
                norm = normalize_material(lr.material, alias_map)
                row["LLaVA_Material"] = lr.material
                row["LLaVA_Material_Norm"] = norm
                row["LLaVA_WWR"] = f"{lr.wwr:.3f}" if lr.wwr >= 0 else "N/A"
                row["LLaVA_Match"] = "Yes" if norm in gt_mats else "No"

            # WWR detection models
            for name in wwr_model_names:
                model_results = wwr_model_results[name]
                if record.image_id in model_results:
                    wr = model_results[record.image_id]
                    row[f"{name}_WWR"] = f"{wr.wwr:.3f}"
                    row[f"{name}_Detections"] = str(wr.num_detections)
                    row[f"{name}_Method"] = wr.method

            writer.writerow(row)

    logger.info("Detailed results saved to %s", output_path)


def _pct_column(label: str) -> str:
    """Canonical column name for a material's percentage in the full-results CSV."""
    return label.replace(" ", "_") + "_Pct"


def save_full_results_csv(
    output_path: str,
    image_records: list[BuildingRecord],
    model_results: dict,
    alias_map: dict[str, str],
    labels: list[str],
    include_ground_truth: bool,
):
    """Write a YOLO-style per-building CSV: dominant material + WWR + per-label %.

    `model_results` maps image_id -> result object with fields front_wall,
    sec_front_wall, proportion, wwr. When `include_ground_truth` is True,
    extra columns (GT_Front_Wall, GT_Sec_Front_Wall, GT_Proportion, GT_WWR,
    Match) are appended for curated-dataset inspection."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pct_cols = [_pct_column(label) for label in labels]
    fieldnames = ["Image_ID", "Address", "Dominant_Material", "WWR"] + pct_cols
    if include_ground_truth:
        fieldnames += [
            "GT_Front_Wall", "GT_Sec_Front_Wall", "GT_Proportion",
            "GT_WWR", "Match",
        ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in image_records:
            if record.image_id not in model_results:
                continue
            result = model_results[record.image_id]

            ratios = compute_ratios(
                result.front_wall,
                getattr(result, "sec_front_wall", "") or "",
                getattr(result, "proportion", "1") or "1",
                alias_map,
                labels,
            )
            dominant = dominant_material(ratios)
            wwr = getattr(result, "wwr", -1.0)

            row = {
                "Image_ID": record.image_id,
                "Address": record.address,
                "Dominant_Material": dominant or "Unknown",
                "WWR": f"{wwr:.3f}" if wwr >= 0 else "N/A",
            }
            for label, col in zip(labels, pct_cols):
                row[col] = f"{ratios[label] * 100.0:.1f}"

            if include_ground_truth:
                gt_mats = ground_truth_materials(record)
                row["GT_Front_Wall"] = record.front_wall
                row["GT_Sec_Front_Wall"] = record.sec_front_wall
                row["GT_Proportion"] = record.proportion
                gt_wwr = getattr(record, "wwr", "")
                row["GT_WWR"] = gt_wwr if gt_wwr else ""
                row["Match"] = "Yes" if dominant and dominant in gt_mats else "No"

            writer.writerow(row)

    logger.info("Full-results CSV saved to %s", output_path)


def format_confusion_matrix(confusion: dict[str, dict[str, int]], labels: list[str]) -> str:
    """Format confusion matrix as aligned text table."""
    # Abbreviate labels for display
    abbr = {
        "Brick": "Brck",
        "Stucco": "Stuc",
        "Vinyl": "Vnyl",
        "Decorative stone": "DStn",
        "Aluminum Composite": "AlCm",
        "Metal": "Metl",
        "Fibercement": "FbCm",
    }
    short = [abbr.get(l, l[:4]) for l in labels]

    header = "GT\\Pred  " + "  ".join(f"{s:>5}" for s in short)
    lines = [header, "-" * len(header)]
    for i, label in enumerate(labels):
        row_vals = [confusion.get(label, {}).get(pl, 0) for pl in labels]
        row_str = f"{short[i]:<9}" + "  ".join(f"{v:>5}" for v in row_vals)
        lines.append(row_str)
    return "\n".join(lines)


def save_summary_report(
    output_path: str,
    material_labels: list[str],
    claude_metrics: MaterialMetrics | None = None,
    clip_metrics: MaterialMetrics | None = None,
    llava_metrics: MaterialMetrics | None = None,
    claude_wwr: WWRStats | None = None,
    llava_wwr: WWRStats | None = None,
    wwr_model_stats: dict[str, WWRStats] | None = None,
):
    """Save summary report with per-model comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("ZERO-SHOT BUILDING FACADE ANALYSIS - SUMMARY REPORT")
    lines.append("=" * 70)

    # Comparison table
    lines.append("\n--- MATERIAL CLASSIFICATION: MODEL COMPARISON ---\n")
    header = f"{'Metric':<25}  "
    models = []
    metrics_list = []
    if claude_metrics:
        models.append("Claude")
        metrics_list.append(claude_metrics)
    if clip_metrics:
        models.append("CLIP")
        metrics_list.append(clip_metrics)
    if llava_metrics:
        models.append("LLaVA")
        metrics_list.append(llava_metrics)

    header += "  ".join(f"{m:>10}" for m in models)
    lines.append(header)
    lines.append("-" * len(header))

    for metric_name, getter in [
        ("Accuracy", lambda m: f"{m.accuracy:.3f}"),
        ("Correct / Total", lambda m: f"{m.correct}/{m.total}"),
        ("Macro Precision", lambda m: f"{m.macro_precision:.3f}"),
        ("Macro Recall", lambda m: f"{m.macro_recall:.3f}"),
        ("Macro F1", lambda m: f"{m.macro_f1:.3f}"),
        ("Weighted F1", lambda m: f"{m.weighted_f1:.3f}"),
    ]:
        row = f"{metric_name:<25}  "
        row += "  ".join(f"{getter(m):>10}" for m in metrics_list)
        lines.append(row)

    # Per-class breakdown for each model
    for model_name, metrics in zip(models, metrics_list):
        lines.append(f"\n--- {model_name.upper()}: PER-CLASS BREAKDOWN ---\n")
        h = f"{'Class':<25}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}"
        lines.append(h)
        lines.append("-" * len(h))
        for label in material_labels:
            p = metrics.per_class_precision.get(label, 0)
            r = metrics.per_class_recall.get(label, 0)
            f1 = metrics.per_class_f1.get(label, 0)
            lines.append(f"{label:<25}  {p:>10.3f}  {r:>10.3f}  {f1:>10.3f}")

        lines.append(f"\n--- {model_name.upper()}: CONFUSION MATRIX ---\n")
        lines.append(format_confusion_matrix(metrics.confusion_matrix, material_labels))

    # WWR statistics
    wwr_models = []
    wwr_stats = []
    if claude_wwr:
        wwr_models.append("Claude")
        wwr_stats.append(claude_wwr)
    if llava_wwr:
        wwr_models.append("LLaVA")
        wwr_stats.append(llava_wwr)
    if wwr_model_stats:
        for name, stats in wwr_model_stats.items():
            wwr_models.append(name)
            wwr_stats.append(stats)

    if wwr_stats:
        lines.append("\n--- WINDOW-TO-WALL RATIO: PREDICTION STATISTICS ---\n")
        h = f"{'Metric':<25}  " + "  ".join(f"{m:>10}" for m in wwr_models)
        lines.append(h)
        lines.append("-" * len(h))
        for stat_name, getter in [
            ("Valid predictions", lambda s: str(s.count)),
            ("Mean", lambda s: f"{s.mean:.3f}"),
            ("Median", lambda s: f"{s.median:.3f}"),
            ("Std Dev", lambda s: f"{s.std:.3f}"),
            ("Min", lambda s: f"{s.min_val:.3f}"),
            ("Max", lambda s: f"{s.max_val:.3f}"),
        ]:
            row = f"{stat_name:<25}  " + "  ".join(f"{getter(s):>10}" for s in wwr_stats)
            lines.append(row)

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also print to console
    print(report)
    logger.info("Summary report saved to %s", output_path)
