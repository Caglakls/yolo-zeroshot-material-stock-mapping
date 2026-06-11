import logging
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from models.wwr_utils import (
    WWRResult,
    apply_nms,
    compute_wwr_from_masks,
    load_wwr_checkpoint,
    save_wwr_checkpoint,
)

logger = logging.getLogger(__name__)

WINDOW_PROMPT = "window . glass window ."


def detect_bboxes_gdino(
    model,
    processor,
    image_path: str,
    text_prompt: str = WINDOW_PROMPT,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    iou_threshold: float = 0.5,
    device: str = "cuda",
) -> list[tuple[float, float, float, float]]:
    """Run GDINO to get window bounding boxes (pixel coords)."""
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
        bboxes.append(tuple(box.cpu().tolist()))
        scores.append(score.cpu().item())

    if bboxes:
        bboxes, scores = apply_nms(bboxes, scores, iou_threshold)

    return bboxes


def segment_with_sam(
    sam_model: SamModel,
    sam_processor: SamProcessor,
    image_path: str,
    bboxes: list[tuple[float, float, float, float]],
    device: str = "cuda",
) -> np.ndarray | None:
    """Segment detected windows using SAM. Returns union mask [H, W] or None."""
    if not bboxes:
        return None

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # SAM expects boxes as [[x1, y1, x2, y2], ...]
    input_boxes = [list(b) for b in bboxes]

    inputs = sam_processor(
        image,
        input_boxes=[input_boxes],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    # outputs.pred_masks shape: [batch, num_boxes, num_masks, H, W]
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]  # first batch item

    # masks shape: [num_boxes, num_masks, H, W] -- take best mask per box (index 0)
    if masks.ndim == 4:
        best_masks = masks[:, 0, :, :]  # [num_boxes, H, W]
    elif masks.ndim == 3:
        best_masks = masks
    else:
        return None

    # Union all masks
    union = np.logical_or.reduce(best_masks.numpy().astype(bool), axis=0)
    return union


def run_gdino_sam_batch(
    image_records: list,
    gdino_model_name: str,
    sam_model_name: str,
    device: str = "cuda",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    checkpoint_dir: str = "results",
    resume: bool = False,
) -> dict[str, WWRResult]:
    """Two-phase pipeline: GDINO detection then SAM segmentation.
    Loads models sequentially to fit 8GB VRAM."""
    checkpoint_path = os.path.join(checkpoint_dir, "gdino_sam_checkpoint.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = load_wwr_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(results.keys())

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    if not remaining:
        return results

    # Phase 1: GDINO detection -- collect bboxes for all images
    logger.info("Phase 1: Running GDINO detection...")
    gdino_processor = AutoProcessor.from_pretrained(gdino_model_name)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gdino_model_name
    ).to(device)
    gdino_model.eval()

    all_bboxes = {}
    for record in tqdm(remaining, desc="GDINO+SAM Phase 1 (detection)"):
        try:
            bboxes = detect_bboxes_gdino(
                gdino_model, gdino_processor, record.image_path,
                box_threshold=box_threshold, text_threshold=text_threshold,
                device=device,
            )
            all_bboxes[record.image_id] = bboxes
        except Exception as e:
            logger.error("GDINO detection failed for %s: %s", record.image_id, e)
            all_bboxes[record.image_id] = []

    del gdino_model, gdino_processor
    torch.cuda.empty_cache()

    # Phase 2: SAM segmentation
    logger.info("Phase 2: Running SAM segmentation...")
    sam_processor = SamProcessor.from_pretrained(sam_model_name)
    sam_model = SamModel.from_pretrained(sam_model_name).to(device)
    sam_model.eval()

    for i, record in enumerate(tqdm(remaining, desc="GDINO+SAM Phase 2 (segmentation)")):
        bboxes = all_bboxes.get(record.image_id, [])
        try:
            image = Image.open(record.image_path).convert("RGB")
            img_w, img_h = image.size

            if not bboxes:
                results[record.image_id] = WWRResult(
                    wwr=0.0, num_detections=0, total_window_area=0.0,
                    image_area=float(img_w * img_h), method="segmentation",
                )
            else:
                union_mask = segment_with_sam(
                    sam_model, sam_processor, record.image_path, bboxes, device
                )
                if union_mask is not None:
                    results[record.image_id] = compute_wwr_from_masks(
                        union_mask, img_w, img_h
                    )
                    results[record.image_id] = WWRResult(
                        wwr=results[record.image_id].wwr,
                        num_detections=len(bboxes),
                        total_window_area=results[record.image_id].total_window_area,
                        image_area=results[record.image_id].image_area,
                        method="segmentation",
                    )
                else:
                    results[record.image_id] = compute_wwr_from_masks(
                        np.array([]), img_w, img_h
                    )
        except Exception as e:
            logger.error("SAM segmentation failed for %s: %s", record.image_id, e)
            results[record.image_id] = WWRResult(
                wwr=0.0, num_detections=0, total_window_area=0.0,
                image_area=0.0, method="segmentation",
            )

        if (i + 1) % 50 == 0:
            save_wwr_checkpoint(checkpoint_path, results)

    save_wwr_checkpoint(checkpoint_path, results)

    del sam_model, sam_processor
    torch.cuda.empty_cache()

    logger.info("GDINO+SAM: processed %d images", len(results))
    return results
