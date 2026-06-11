import csv
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BuildingRecord:
    image_id: str
    image_path: Optional[str]
    address: str
    front_wall: str
    sec_front_wall: str
    back_wall: str
    proportion: str
    building_type: str
    approx_hgt: str
    number_stories: str
    year_built: str
    wwr: str = ""


def load_ground_truth(csv_path: str) -> dict[str, BuildingRecord]:
    """Parse CSV and return dict keyed by Image_ID."""
    records = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("Image_ID", "").strip()
            if not image_id:
                continue
            records[image_id] = BuildingRecord(
                image_id=image_id,
                image_path=None,
                address=row.get("Address", "").strip(),
                front_wall=row.get("Front_wall", "").strip(),
                sec_front_wall=row.get("Sec_front_wall", "").strip(),
                back_wall=row.get("Back_wall", "").strip(),
                proportion=row.get("Proportion", "").strip(),
                building_type=row.get("Type", "").strip(),
                approx_hgt=row.get("approx_hgt", "").strip(),
                number_stories=row.get("number_stories", "").strip(),
                year_built=row.get("year_built", "").strip(),
                wwr=row.get("WWR", "").strip(),
            )
    logger.info("Loaded %d records from CSV", len(records))
    return records


def build_image_index(image_dir: str) -> dict[str, str]:
    """Scan image directory, return dict mapping stripped filename (no ext) to full path."""
    index = {}
    for fname in os.listdir(image_dir):
        lower = fname.lower()
        if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
            continue
        stem = os.path.splitext(fname)[0].strip()
        full_path = os.path.join(image_dir, fname)
        index[stem] = full_path
    logger.info("Indexed %d image files", len(index))
    return index


def match_records_to_images(
    records: dict[str, BuildingRecord],
    image_index: dict[str, str],
    corrections: dict[str, str],
) -> list[BuildingRecord]:
    """Match CSV records to image files. Returns list of matched BuildingRecords."""
    matched = []
    unmatched = []

    for image_id, record in records.items():
        # Try exact match
        if image_id in image_index:
            record.image_path = image_index[image_id]
            matched.append(record)
            continue

        # Try correction map
        corrected = corrections.get(image_id)
        if corrected and corrected in image_index:
            record.image_path = image_index[corrected]
            matched.append(record)
            continue

        unmatched.append(image_id)

    if unmatched:
        logger.warning("Unmatched CSV records (%d): %s", len(unmatched), unmatched)
    logger.info("Matched %d / %d records to images", len(matched), len(records))
    return matched


def normalize_material(raw: str, alias_map: dict[str, str]) -> str:
    """Normalize a model's raw material prediction to a canonical label."""
    cleaned = raw.strip().lower()
    if not cleaned:
        return raw

    # Exact alias match
    if cleaned in alias_map:
        return alias_map[cleaned]

    # Check if any alias key is a substring of the prediction
    for alias_key, canonical in alias_map.items():
        if alias_key in cleaned:
            return canonical

    # Check if the prediction is a substring of any alias key
    for alias_key, canonical in alias_map.items():
        if cleaned in alias_key:
            return canonical

    return raw.strip()


def ground_truth_materials(record: BuildingRecord) -> list[str]:
    """Return list of acceptable ground truth materials (non-empty only)."""
    materials = []
    if record.front_wall:
        materials.append(record.front_wall)
    if record.sec_front_wall:
        materials.append(record.sec_front_wall)
    return materials
