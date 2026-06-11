"""Preprocess raw building photos: detect facade with Grounding DINO and crop."""

import argparse
import json
import logging
import os
import shutil

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from models.wwr_utils import apply_nms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BUILDING_PROMPT = "building . building facade . house front ."
INPUT_DIR = "new_data"
OUTPUT_DIR = "new_data_cropped"


def detect_building(model, processor, image_path, box_threshold=0.20,
                    text_threshold=0.20, device="cuda"):
    """Detect building facades in an image, return best bbox or None."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    inputs = processor(images=image, text=BUILDING_PROMPT,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(img_h, img_w)],
    )[0]

    bboxes = []
    scores = []
    for box, score in zip(results["boxes"], results["scores"]):
        xyxy = box.cpu().tolist()
        bboxes.append(tuple(xyxy))
        scores.append(score.cpu().item())

    if not bboxes:
        return None, image

    bboxes, scores = apply_nms(bboxes, scores, iou_threshold=0.5)

    # Select largest bbox by area
    best_idx = 0
    best_area = 0
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_idx = i

    bbox = bboxes[best_idx]
    coverage = best_area / (img_w * img_h)
    return (bbox, coverage), image


def crop_with_padding(image, bbox, padding_ratio=0.10):
    """Crop image to bbox with padding."""
    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))

    return image.crop((cx1, cy1, cx2, cy2))


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw building photos")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image files
    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if args.max_images:
        image_files = image_files[:args.max_images]
    logger.info("Found %d images in %s", len(image_files), args.input_dir)

    # Load GDINO
    model_name = "IDEA-Research/grounding-dino-tiny"
    logger.info("Loading Grounding DINO: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
        args.device
    )
    model.eval()

    detected_count = 0
    fallback_count = 0
    coverages = []

    for fname in tqdm(image_files, desc="Preprocessing"):
        src_path = os.path.join(args.input_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)

        try:
            result, image = detect_building(
                model, processor, src_path,
                box_threshold=args.box_threshold, device=args.device,
            )

            if result is not None:
                bbox, coverage = result
                cropped = crop_with_padding(image, bbox)
                cropped.save(dst_path, quality=95)
                detected_count += 1
                coverages.append(coverage)
            else:
                shutil.copy2(src_path, dst_path)
                fallback_count += 1
        except Exception as e:
            logger.error("Failed on %s: %s", fname, e)
            shutil.copy2(src_path, dst_path)
            fallback_count += 1

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Save log
    log_data = {
        "total_images": len(image_files),
        "detected": detected_count,
        "fallback": fallback_count,
        "avg_coverage": sum(coverages) / len(coverages) if coverages else 0,
        "min_coverage": min(coverages) if coverages else 0,
        "max_coverage": max(coverages) if coverages else 0,
    }
    log_path = os.path.join(args.output_dir, "preprocess_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(
        "Done: %d detected, %d fallback, avg coverage %.3f",
        detected_count, fallback_count,
        log_data["avg_coverage"],
    )


if __name__ == "__main__":
    main()
