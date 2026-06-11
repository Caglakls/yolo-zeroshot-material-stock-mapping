"""Gemini Vision API for building facade material classification and WWR estimation."""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from google import genai
from PIL import Image
from tqdm import tqdm

from models.prompts import DETAILED_PROMPT, parse_detailed_response

logger = logging.getLogger(__name__)


@dataclass
class GeminiResult:
    front_wall: str
    sec_front_wall: str
    proportion: str
    back_wall: str
    wwr: float
    raw_response: str

    @property
    def material(self) -> str:
        return self.front_wall


CHECKPOINT_NAME = "gemini_detailed_checkpoint.json"


def _result_from_raw(raw_text: str) -> GeminiResult:
    parsed = parse_detailed_response(raw_text)
    return GeminiResult(
        front_wall=parsed["front_wall"],
        sec_front_wall=parsed["sec_front_wall"],
        proportion=parsed["proportion"],
        back_wall=parsed["back_wall"],
        wwr=parsed["wwr"],
        raw_response=raw_text,
    )


def query_gemini_single(client: genai.Client, model_name: str,
                        image_path: str) -> GeminiResult:
    """Send one image to Gemini Vision API, parse response."""
    image = Image.open(image_path).convert("RGB")
    response = client.models.generate_content(
        model=model_name,
        contents=[DETAILED_PROMPT, image],
    )
    raw_text = response.text
    return _result_from_raw(raw_text)


def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: str, data: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def _checkpoint_entry_to_result(entry: dict) -> GeminiResult:
    return GeminiResult(
        front_wall=entry.get("front_wall", entry.get("material", "Unknown")),
        sec_front_wall=entry.get("sec_front_wall", "") or "",
        proportion=entry.get("proportion", "1") or "1",
        back_wall=entry.get("back_wall", "") or "",
        wwr=float(entry.get("wwr", -1.0)),
        raw_response=entry.get("raw_response", ""),
    )


def _result_to_checkpoint_entry(result: GeminiResult) -> dict:
    return {
        "front_wall": result.front_wall,
        "sec_front_wall": result.sec_front_wall,
        "proportion": result.proportion,
        "back_wall": result.back_wall,
        "wwr": result.wwr,
        "raw_response": result.raw_response,
    }


def run_gemini_batch(
    image_records: list,
    api_key: str,
    model_name: str,
    requests_per_minute: int = 60,
    checkpoint_dir: str = "results",
    resume: bool = False,
    max_workers: int = 16,
) -> dict[str, GeminiResult]:
    """Process all images through Gemini Vision with concurrent requests."""
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_data = load_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(checkpoint_data.keys())

    client = genai.Client(api_key=api_key)

    results = {}
    for image_id, data in checkpoint_data.items():
        results[image_id] = _checkpoint_entry_to_result(data)

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    lock = threading.Lock()
    done_count = [0]

    def process_one(record):
        try:
            result = query_gemini_single(client, model_name, record.image_path)
            return record.image_id, result, None
        except Exception as e:
            return record.image_id, None, e

    pbar = tqdm(total=len(remaining), desc="Gemini inference")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, r): r for r in remaining}

        for future in as_completed(futures):
            image_id, result, error = future.result()

            with lock:
                if error:
                    logger.error("Gemini failed for %s: %s", image_id, error)
                    result = GeminiResult(
                        front_wall="Unknown", sec_front_wall="", proportion="1",
                        back_wall="", wwr=-1.0, raw_response=str(error),
                    )

                results[image_id] = result
                checkpoint_data[image_id] = _result_to_checkpoint_entry(result)

                done_count[0] += 1
                pbar.update(1)

                if done_count[0] % 50 == 0:
                    save_checkpoint(checkpoint_path, checkpoint_data)
                    logger.info("Checkpoint saved at %d images", len(checkpoint_data))

    pbar.close()
    save_checkpoint(checkpoint_path, checkpoint_data)
    logger.info("Gemini: processed %d images", len(results))
    return results
