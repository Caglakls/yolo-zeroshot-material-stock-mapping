import logging
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from models.wwr_utils import (
    WWRResult,
    apply_nms,
    compute_wwr_from_bboxes,
    load_wwr_checkpoint,
    save_wwr_checkpoint,
)

logger = logging.getLogger(__name__)

WINDOW_PROMPT = "window . glass window ."


def load_grounding_dino(model_name: str, device: str = "cuda"):
    """Load Grounding DINO from HuggingFace transformers."""
    logger.info("Loading Grounding DINO: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def detect_windows_gdino(
    model,
    processor,
    image_path: str,
    text_prompt: str = WINDOW_PROMPT,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    iou_threshold: float = 0.5,
    device: str = "cuda",
) -> WWRResult:
    """Run Grounding DINO on one image, return WWR from bounding boxes."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(img_h, img_w)],
    )[0]

    bboxes = []
    scores = []
    for box, score in zip(results["boxes"], results["scores"]):
        xyxy = box.cpu().tolist()
        bboxes.append(tuple(xyxy))
        scores.append(score.cpu().item())

    # Apply NMS
    if bboxes:
        bboxes, scores = apply_nms(bboxes, scores, iou_threshold)

    return compute_wwr_from_bboxes(bboxes, img_w, img_h)


def run_gdino_batch(
    image_records: list,
    model_name: str,
    device: str = "cuda",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    checkpoint_dir: str = "results",
    resume: bool = False,
) -> dict[str, WWRResult]:
    """Process all images through Grounding DINO."""
    checkpoint_path = os.path.join(checkpoint_dir, "gdino_checkpoint.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = load_wwr_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(results.keys())

    model, processor = load_grounding_dino(model_name, device)

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    for i, record in enumerate(tqdm(remaining, desc="GDINO inference")):
        try:
            result = detect_windows_gdino(
                model, processor, record.image_path,
                box_threshold=box_threshold, text_threshold=text_threshold,
                device=device,
            )
            results[record.image_id] = result
        except Exception as e:
            logger.error("GDINO failed for %s: %s", record.image_id, e)
            results[record.image_id] = WWRResult(
                wwr=0.0, num_detections=0, total_window_area=0.0,
                image_area=0.0, method="bbox",
            )

        if (i + 1) % 50 == 0:
            save_wwr_checkpoint(checkpoint_path, results)

    save_wwr_checkpoint(checkpoint_path, results)

    del model
    torch.cuda.empty_cache()

    logger.info("GDINO: processed %d images", len(results))
    return results
