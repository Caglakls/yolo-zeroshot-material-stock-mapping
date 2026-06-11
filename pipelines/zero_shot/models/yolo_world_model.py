import logging
import os

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLOWorld

from models.wwr_utils import (
    WWRResult,
    apply_nms,
    compute_wwr_from_bboxes,
    load_wwr_checkpoint,
    save_wwr_checkpoint,
)

logger = logging.getLogger(__name__)

WINDOW_CLASSES = ["window", "glass window", "window pane"]


def load_yolo_world(model_size: str = "s"):
    """Load YOLO-World model with open-vocabulary window detection."""
    model_name = f"yolov8{model_size}-worldv2"
    logger.info("Loading YOLO-World model: %s", model_name)
    model = YOLOWorld(model_name)
    model.set_classes(WINDOW_CLASSES)
    return model


def detect_windows_yolo(
    model,
    image_path: str,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
) -> WWRResult:
    """Run YOLO-World on one image, return WWR from bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    results = model.predict(image_path, conf=conf_threshold, verbose=False)

    all_bboxes = []
    all_scores = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().tolist()
            conf = boxes.conf[i].cpu().item()
            all_bboxes.append(tuple(xyxy))
            all_scores.append(conf)

    # Class-agnostic NMS to merge detections across window classes
    if all_bboxes:
        all_bboxes, all_scores = apply_nms(all_bboxes, all_scores, iou_threshold)

    return compute_wwr_from_bboxes(all_bboxes, img_w, img_h)


def run_yolo_world_batch(
    image_records: list,
    model_size: str = "s",
    device: str = "cuda",
    conf_threshold: float = 0.15,
    checkpoint_dir: str = "results",
    resume: bool = False,
) -> dict[str, WWRResult]:
    """Process all images through YOLO-World."""
    checkpoint_path = os.path.join(checkpoint_dir, "yolo_world_checkpoint.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = load_wwr_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(results.keys())

    model = load_yolo_world(model_size)

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    for i, record in enumerate(tqdm(remaining, desc="YOLO-World inference")):
        try:
            result = detect_windows_yolo(model, record.image_path, conf_threshold)
            results[record.image_id] = result
        except Exception as e:
            logger.error("YOLO-World failed for %s: %s", record.image_id, e)
            results[record.image_id] = WWRResult(
                wwr=0.0, num_detections=0, total_window_area=0.0,
                image_area=0.0, method="bbox",
            )

        if (i + 1) % 50 == 0:
            save_wwr_checkpoint(checkpoint_path, results)

    save_wwr_checkpoint(checkpoint_path, results)
    logger.info("YOLO-World: processed %d images", len(results))
    return results
