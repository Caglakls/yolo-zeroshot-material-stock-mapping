import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)


@dataclass
class WWRResult:
    wwr: float  # 0.0 to 1.0
    num_detections: int
    total_window_area: float  # pixels
    image_area: float  # pixels
    method: str  # "bbox" or "segmentation"


def compute_wwr_from_bboxes(
    bboxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> WWRResult:
    """Compute WWR from bounding boxes (x1, y1, x2, y2 in pixels)."""
    image_area = float(img_w * img_h)
    if not bboxes or image_area == 0:
        return WWRResult(wwr=0.0, num_detections=0, total_window_area=0.0,
                         image_area=image_area, method="bbox")

    total_area = 0.0
    for x1, y1, x2, y2 in bboxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        total_area += w * h

    wwr = min(total_area / image_area, 1.0)
    return WWRResult(wwr=wwr, num_detections=len(bboxes),
                     total_window_area=total_area, image_area=image_area,
                     method="bbox")


def compute_wwr_from_masks(
    masks: np.ndarray,
    img_w: int,
    img_h: int,
) -> WWRResult:
    """Compute WWR from binary masks. masks shape: [N, H, W] or [H, W]."""
    image_area = float(img_w * img_h)
    if masks is None or masks.size == 0 or image_area == 0:
        return WWRResult(wwr=0.0, num_detections=0, total_window_area=0.0,
                         image_area=image_area, method="segmentation")

    if masks.ndim == 2:
        union_mask = masks.astype(bool)
        n_detections = 1
    else:
        union_mask = np.logical_or.reduce(masks.astype(bool), axis=0)
        n_detections = masks.shape[0]

    total_pixels = float(union_mask.sum())
    wwr = min(total_pixels / image_area, 1.0)
    return WWRResult(wwr=wwr, num_detections=n_detections,
                     total_window_area=total_pixels, image_area=image_area,
                     method="segmentation")


def apply_nms(
    bboxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Class-agnostic NMS using torchvision."""
    if not bboxes:
        return [], []

    boxes_t = torch.tensor(bboxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    keep = keep.tolist()

    return [bboxes[i] for i in keep], [scores[i] for i in keep]


def load_wwr_checkpoint(checkpoint_path: str) -> dict[str, WWRResult]:
    """Load WWR checkpoint from JSON."""
    if not os.path.exists(checkpoint_path):
        return {}
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    results = {}
    for image_id, d in data.items():
        results[image_id] = WWRResult(
            wwr=d["wwr"],
            num_detections=d["num_detections"],
            total_window_area=d["total_window_area"],
            image_area=d["image_area"],
            method=d["method"],
        )
    return results


def save_wwr_checkpoint(checkpoint_path: str, results: dict[str, WWRResult]):
    """Save WWR results checkpoint to JSON."""
    data = {}
    for image_id, r in results.items():
        data[image_id] = {
            "wwr": r.wwr,
            "num_detections": r.num_detections,
            "total_window_area": r.total_window_area,
            "image_area": r.image_area,
            "method": r.method,
        }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)
