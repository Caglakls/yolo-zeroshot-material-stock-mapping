"""Prediction-only pipeline for new (preprocessed) building images without ground truth."""

import argparse
import logging
import os
from collections import Counter

import config
from data_loader import BuildingRecord, build_image_index, normalize_material
from evaluation import compute_wwr_stats, save_full_results_csv
from materials_utils import compute_ratios, dominant_material

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NEW_IMAGE_DIR = os.path.join(config.PROJECT_DIR, "new_data_cropped")
NEW_RESULTS_DIR = os.path.join(config.PROJECT_DIR, "results_new_data")


def build_records_from_index(image_index):
    """Create BuildingRecord objects from image index (no ground truth)."""
    records = []
    for stem, path in sorted(image_index.items()):
        records.append(BuildingRecord(
            image_id=stem,
            image_path=path,
            address=stem,
            front_wall="",
            sec_front_wall="",
            back_wall="",
            proportion="",
            building_type="",
            approx_hgt="",
            number_stories="",
            year_built="",
        ))
    return records


def save_prediction_summary(output_path, material_labels, alias_map,
                            vlm_results_by_model=None,
                            vlm_wwr_stats=None,
                            wwr_model_stats=None):
    """Summary text with per-model dominant-material distribution, mean
    per-material ratio, and WWR statistics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("NEW DATA - ZERO-SHOT PREDICTION SUMMARY")
    lines.append("=" * 70)

    for model_name, results in (vlm_results_by_model or {}).items():
        if not results:
            continue
        lines.append(f"\n--- {model_name.upper()}: DOMINANT MATERIAL DISTRIBUTION ---\n")

        dominant_counter = Counter()
        ratio_sums = {label: 0.0 for label in material_labels}

        for r in results.values():
            ratios = compute_ratios(
                r.front_wall,
                getattr(r, "sec_front_wall", "") or "",
                getattr(r, "proportion", "1") or "1",
                alias_map,
                material_labels,
            )
            dom = dominant_material(ratios) or "Unknown"
            dominant_counter[dom] += 1
            for label in material_labels:
                ratio_sums[label] += ratios[label]

        total = sum(dominant_counter.values())
        lines.append(f"{'Material':<25}  {'Count':>8}  {'Percent':>8}  {'AvgRatio%':>10}")
        lines.append("-" * 60)
        for label in material_labels:
            c = dominant_counter.get(label, 0)
            pct = 100.0 * c / total if total else 0
            avg = 100.0 * ratio_sums[label] / total if total else 0
            lines.append(f"{label:<25}  {c:>8}  {pct:>7.1f}%  {avg:>9.1f}%")
        other = {k: v for k, v in dominant_counter.items() if k not in material_labels}
        for k, v in sorted(other.items(), key=lambda x: -x[1]):
            pct = 100.0 * v / total if total else 0
            lines.append(f"{k:<25}  {v:>8}  {pct:>7.1f}%")
        lines.append(f"{'TOTAL':<25}  {total:>8}")

    wwr_models = []
    wwr_stats_list = []
    for name, stats in (vlm_wwr_stats or {}).items():
        wwr_models.append(name)
        wwr_stats_list.append(stats)
    for name, stats in (wwr_model_stats or {}).items():
        wwr_models.append(name)
        wwr_stats_list.append(stats)

    if wwr_stats_list:
        lines.append("\n--- WINDOW-TO-WALL RATIO: PREDICTION STATISTICS ---\n")
        h = f"{'Metric':<25}  " + "  ".join(f"{m:>14}" for m in wwr_models)
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
            row = f"{stat_name:<25}  " + "  ".join(f"{getter(s):>14}" for s in wwr_stats_list)
            lines.append(row)

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    logger.info("Summary saved to %s", output_path)


MODEL_SLUG = {
    config.CLAUDE_DISPLAY: "claude",
    config.GPT_DISPLAY: "gpt",
    config.GEMINI_DISPLAY: "gemini",
}


def main():
    parser = argparse.ArgumentParser(description="Run models on new building images (no ground truth)")
    parser.add_argument(
        "--models", nargs="+",
        default=["claude", "gpt", "gemini"],
        choices=["claude", "gpt", "gemini", "clip", "yolo_world", "gdino", "gdino_sam"],
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-dir", default=NEW_IMAGE_DIR)
    parser.add_argument("--results-dir", default=NEW_RESULTS_DIR)
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip rendering overview PNG figures")
    args = parser.parse_args()

    image_index = build_image_index(args.image_dir)
    records = build_records_from_index(image_index)
    if args.max_images:
        records = records[:args.max_images]
    logger.info("Processing %d images from %s", len(records), args.image_dir)

    os.makedirs(args.results_dir, exist_ok=True)

    vlm_results_by_model: dict[str, dict] = {}

    if "claude" in args.models:
        logger.info("--- Running Claude Vision ---")
        if not config.ANTHROPIC_BASE_URL or not config.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY must be set. Skipping.")
        else:
            from models.claude_vision import run_claude_batch
            vlm_results_by_model[config.CLAUDE_DISPLAY] = run_claude_batch(
                records,
                base_url=config.ANTHROPIC_BASE_URL,
                api_key=config.ANTHROPIC_API_KEY,
                model=config.CLAUDE_MODEL,
                requests_per_minute=config.CLAUDE_REQUESTS_PER_MINUTE,
                checkpoint_dir=args.results_dir,
                resume=args.resume,
            )

    if "gpt" in args.models:
        logger.info("--- Running GPT-4o Vision ---")
        if not config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY must be set. Skipping.")
        else:
            from models.gpt_vision import run_gpt_batch
            vlm_results_by_model[config.GPT_DISPLAY] = run_gpt_batch(
                records,
                api_key=config.OPENAI_API_KEY,
                model=config.GPT_MODEL,
                requests_per_minute=config.GPT_REQUESTS_PER_MINUTE,
                checkpoint_dir=args.results_dir,
                resume=args.resume,
            )

    if "gemini" in args.models:
        logger.info("--- Running Gemini Vision ---")
        if not config.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY must be set. Skipping.")
        else:
            from models.gemini_vision import run_gemini_batch
            vlm_results_by_model[config.GEMINI_DISPLAY] = run_gemini_batch(
                records,
                api_key=config.GOOGLE_API_KEY,
                model_name=config.GEMINI_MODEL,
                requests_per_minute=config.GEMINI_REQUESTS_PER_MINUTE,
                checkpoint_dir=args.results_dir,
                resume=args.resume,
            )

    wwr_model_stats = {}
    if "yolo_world" in args.models:
        logger.info("--- Running YOLO-World ---")
        from models.yolo_world_model import run_yolo_world_batch
        yolo_results = run_yolo_world_batch(
            records, model_size=config.YOLO_WORLD_MODEL_SIZE, device=args.device,
            conf_threshold=config.YOLO_WORLD_CONF_THRESHOLD,
            checkpoint_dir=args.results_dir, resume=args.resume,
        )
        wwr_model_stats[config.YOLO_WORLD_DISPLAY] = compute_wwr_stats(
            {k: v.wwr for k, v in yolo_results.items()}
        )

    if "gdino" in args.models:
        logger.info("--- Running Grounding DINO ---")
        from models.gdino_model import run_gdino_batch
        gdino_results = run_gdino_batch(
            records, model_name=config.GDINO_MODEL_NAME, device=args.device,
            box_threshold=config.GDINO_BOX_THRESHOLD,
            text_threshold=config.GDINO_TEXT_THRESHOLD,
            checkpoint_dir=args.results_dir, resume=args.resume,
        )
        wwr_model_stats[config.GDINO_DISPLAY] = compute_wwr_stats(
            {k: v.wwr for k, v in gdino_results.items()}
        )

    if "gdino_sam" in args.models:
        logger.info("--- Running Grounding DINO + SAM ---")
        from models.gdino_sam_model import run_gdino_sam_batch
        gdino_sam_results = run_gdino_sam_batch(
            records, gdino_model_name=config.GDINO_MODEL_NAME,
            sam_model_name=config.SAM_MODEL_NAME, device=args.device,
            box_threshold=config.GDINO_BOX_THRESHOLD,
            text_threshold=config.GDINO_TEXT_THRESHOLD,
            checkpoint_dir=args.results_dir, resume=args.resume,
        )
        wwr_model_stats[config.GDINO_SAM_DISPLAY] = compute_wwr_stats(
            {k: v.wwr for k, v in gdino_sam_results.items()}
        )

    # Per-model raw_gsv_results_{model}.csv (YOLO-comparable schema)
    vlm_wwr_stats = {}
    for model_name, results in vlm_results_by_model.items():
        if not results:
            continue
        slug = MODEL_SLUG.get(model_name, model_name.lower().replace("-", "_"))
        out_path = os.path.join(args.results_dir, f"raw_gsv_results_{slug}.csv")
        save_full_results_csv(
            out_path, records, results,
            config.MATERIAL_ALIASES, config.MATERIAL_LABELS,
            include_ground_truth=False,
        )
        vlm_wwr_stats[model_name] = compute_wwr_stats(
            {k: v.wwr for k, v in results.items()}
        )
        if not args.no_overview:
            from visualize import render_overview
            png_path = os.path.join(args.results_dir, f"overview_{slug}.png")
            render_overview(out_path, png_path,
                            f"{model_name} - Raw GSV ({len(records)} buildings)")

    summary_path = os.path.join(args.results_dir, "summary_stats.txt")
    save_prediction_summary(
        summary_path, config.MATERIAL_LABELS, config.MATERIAL_ALIASES,
        vlm_results_by_model=vlm_results_by_model,
        vlm_wwr_stats=vlm_wwr_stats,
        wwr_model_stats=wwr_model_stats or None,
    )

    logger.info("All done. Results saved to %s", args.results_dir)


if __name__ == "__main__":
    main()
