import argparse
import logging
import os

import config
from data_loader import (
    build_image_index,
    load_ground_truth,
    match_records_to_images,
)
from evaluation import (
    compute_wwr_stats,
    evaluate_materials,
    save_detailed_results,
    save_full_results_csv,
    save_summary_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_SLUG = {
    config.CLAUDE_DISPLAY: "claude",
    config.GPT_DISPLAY: "gpt",
    config.GEMINI_DISPLAY: "gemini",
}


def main():
    parser = argparse.ArgumentParser(description="Zero-shot building facade analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude", "gpt", "gemini"],
        choices=["claude", "gpt", "gemini", "clip", "llava",
                 "yolo_world", "gdino", "gdino_sam"],
        help="Which models to run (default: claude gpt gemini)",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip rendering overview PNG figures")
    args = parser.parse_args()

    # 1. Load data
    logger.info("Loading ground truth from %s", config.CSV_PATH)
    records = load_ground_truth(config.CSV_PATH)
    image_index = build_image_index(config.IMAGE_DIR)
    matched = match_records_to_images(records, image_index, config.FILENAME_CORRECTIONS)

    if args.max_images:
        matched = matched[: args.max_images]
        logger.info("Limited to %d images for testing", len(matched))

    logger.info("Processing %d images", len(matched))

    records_by_id = {r.image_id: r for r in matched}
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 2. Run models
    vlm_results: dict[str, dict] = {}
    clip_results = None
    llava_results = None

    if "clip" in args.models:
        logger.info("--- Running CLIP ---")
        from models.clip_model import run_clip_batch
        clip_results = run_clip_batch(
            matched, config.CLIP_MODEL_NAME, device=args.device,
        )

    if "claude" in args.models:
        logger.info("--- Running Claude Vision ---")
        if not config.ANTHROPIC_BASE_URL or not config.ANTHROPIC_API_KEY:
            logger.error("ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY must be set. Skipping.")
        else:
            from models.claude_vision import run_claude_batch
            vlm_results[config.CLAUDE_DISPLAY] = run_claude_batch(
                matched,
                base_url=config.ANTHROPIC_BASE_URL,
                api_key=config.ANTHROPIC_API_KEY,
                model=config.CLAUDE_MODEL,
                requests_per_minute=config.CLAUDE_REQUESTS_PER_MINUTE,
                checkpoint_dir=config.RESULTS_DIR,
                resume=args.resume,
            )

    if "gpt" in args.models:
        logger.info("--- Running GPT-4o Vision ---")
        if not config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY must be set. Skipping.")
        else:
            from models.gpt_vision import run_gpt_batch
            vlm_results[config.GPT_DISPLAY] = run_gpt_batch(
                matched,
                api_key=config.OPENAI_API_KEY,
                model=config.GPT_MODEL,
                requests_per_minute=config.GPT_REQUESTS_PER_MINUTE,
                checkpoint_dir=config.RESULTS_DIR,
                resume=args.resume,
            )

    if "gemini" in args.models:
        logger.info("--- Running Gemini Vision ---")
        if not config.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY must be set. Skipping.")
        else:
            from models.gemini_vision import run_gemini_batch
            vlm_results[config.GEMINI_DISPLAY] = run_gemini_batch(
                matched,
                api_key=config.GOOGLE_API_KEY,
                model_name=config.GEMINI_MODEL,
                requests_per_minute=config.GEMINI_REQUESTS_PER_MINUTE,
                checkpoint_dir=config.RESULTS_DIR,
                resume=args.resume,
            )

    if "llava" in args.models:
        logger.info("--- Running LLaVA ---")
        from models.llava_model import run_llava_batch
        llava_results = run_llava_batch(
            matched, config.LLAVA_MODEL_NAME, device=args.device,
            checkpoint_dir=config.RESULTS_DIR, resume=args.resume,
        )

    # 2b. WWR detection models
    yolo_world_results = None
    gdino_results = None
    gdino_sam_results = None

    if "yolo_world" in args.models:
        logger.info("--- Running YOLO-World ---")
        from models.yolo_world_model import run_yolo_world_batch
        yolo_world_results = run_yolo_world_batch(
            matched, model_size=config.YOLO_WORLD_MODEL_SIZE, device=args.device,
            conf_threshold=config.YOLO_WORLD_CONF_THRESHOLD,
            checkpoint_dir=config.RESULTS_DIR, resume=args.resume,
        )

    if "gdino" in args.models:
        logger.info("--- Running Grounding DINO ---")
        from models.gdino_model import run_gdino_batch
        gdino_results = run_gdino_batch(
            matched, model_name=config.GDINO_MODEL_NAME, device=args.device,
            box_threshold=config.GDINO_BOX_THRESHOLD,
            text_threshold=config.GDINO_TEXT_THRESHOLD,
            checkpoint_dir=config.RESULTS_DIR, resume=args.resume,
        )

    if "gdino_sam" in args.models:
        logger.info("--- Running Grounding DINO + SAM ---")
        from models.gdino_sam_model import run_gdino_sam_batch
        gdino_sam_results = run_gdino_sam_batch(
            matched, gdino_model_name=config.GDINO_MODEL_NAME,
            sam_model_name=config.SAM_MODEL_NAME, device=args.device,
            box_threshold=config.GDINO_BOX_THRESHOLD,
            text_threshold=config.GDINO_TEXT_THRESHOLD,
            checkpoint_dir=config.RESULTS_DIR, resume=args.resume,
        )

    # 3. Evaluate (existing classification + WWR stats pipeline)
    logger.info("--- Evaluating results ---")

    per_model_metrics = {}
    per_model_wwr_stats = {}

    for model_name, results in vlm_results.items():
        preds = {k: v.material for k, v in results.items()}
        per_model_metrics[model_name] = evaluate_materials(
            preds, records_by_id, config.MATERIAL_ALIASES, config.MATERIAL_LABELS,
        )
        per_model_wwr_stats[model_name] = compute_wwr_stats(
            {k: v.wwr for k, v in results.items()}
        )

    clip_metrics = None
    if clip_results:
        clip_preds = {k: v.material for k, v in clip_results.items()}
        clip_metrics = evaluate_materials(
            clip_preds, records_by_id, config.MATERIAL_ALIASES, config.MATERIAL_LABELS,
        )

    llava_metrics = None
    llava_wwr_stats = None
    if llava_results:
        llava_preds = {k: v.material for k, v in llava_results.items()}
        llava_metrics = evaluate_materials(
            llava_preds, records_by_id, config.MATERIAL_ALIASES, config.MATERIAL_LABELS,
        )
        llava_wwr_stats = compute_wwr_stats(
            {k: v.wwr for k, v in llava_results.items()}
        )

    wwr_model_results = {}
    wwr_model_stats = {}
    if yolo_world_results:
        wwr_model_results["YOLO_World"] = yolo_world_results
        wwr_model_stats["YOLO_World"] = compute_wwr_stats(
            {k: v.wwr for k, v in yolo_world_results.items()}
        )
    if gdino_results:
        wwr_model_results["GDINO"] = gdino_results
        wwr_model_stats["GDINO"] = compute_wwr_stats(
            {k: v.wwr for k, v in gdino_results.items()}
        )
    if gdino_sam_results:
        wwr_model_results["GDINO_SAM"] = gdino_sam_results
        wwr_model_stats["GDINO_SAM"] = compute_wwr_stats(
            {k: v.wwr for k, v in gdino_sam_results.items()}
        )

    # 4. Save legacy combined CSV + summary report
    detailed_path = os.path.join(config.RESULTS_DIR, "detailed_results.csv")
    save_detailed_results(
        detailed_path, matched,
        vlm_results.get(config.CLAUDE_DISPLAY),
        clip_results,
        llava_results,
        config.MATERIAL_ALIASES,
        wwr_model_results=wwr_model_results or None,
    )

    summary_path = os.path.join(config.RESULTS_DIR, "summary_report.txt")
    save_summary_report(
        summary_path, config.MATERIAL_LABELS,
        claude_metrics=per_model_metrics.get(config.CLAUDE_DISPLAY),
        clip_metrics=clip_metrics,
        llava_metrics=llava_metrics,
        claude_wwr=per_model_wwr_stats.get(config.CLAUDE_DISPLAY),
        llava_wwr=llava_wwr_stats,
        wwr_model_stats=wwr_model_stats or None,
    )

    # 5. Write per-model YOLO-comparable full-results CSV + overview PNG
    for model_name, results in vlm_results.items():
        slug = MODEL_SLUG.get(model_name, model_name.lower().replace("-", "_"))
        csv_path = os.path.join(config.RESULTS_DIR, f"curated_full_results_{slug}.csv")
        save_full_results_csv(
            csv_path, matched, results,
            config.MATERIAL_ALIASES, config.MATERIAL_LABELS,
            include_ground_truth=True,
        )
        if not args.no_overview:
            from visualize import render_overview
            png_path = os.path.join(config.RESULTS_DIR, f"overview_{slug}.png")
            render_overview(csv_path, png_path, f"{model_name} - Curated ({len(matched)} buildings)")

    logger.info("All done. Results saved to %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
