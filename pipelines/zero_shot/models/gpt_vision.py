"""GPT-4o Vision API for building facade material classification and WWR estimation."""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass

import openai
from tqdm import tqdm

from models.prompts import DETAILED_PROMPT, parse_detailed_response

logger = logging.getLogger(__name__)


@dataclass
class GPTResult:
    front_wall: str
    sec_front_wall: str
    proportion: str
    back_wall: str
    wwr: float
    raw_response: str

    @property
    def material(self) -> str:
        return self.front_wall


CHECKPOINT_NAME = "gpt_detailed_checkpoint.json"


def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file, return (base64_data, media_type)."""
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _result_from_raw(raw_text: str) -> GPTResult:
    parsed = parse_detailed_response(raw_text)
    return GPTResult(
        front_wall=parsed["front_wall"],
        sec_front_wall=parsed["sec_front_wall"],
        proportion=parsed["proportion"],
        back_wall=parsed["back_wall"],
        wwr=parsed["wwr"],
        raw_response=raw_text,
    )


def query_gpt_single(client: openai.OpenAI, image_path: str, model: str) -> GPTResult:
    """Send one image to GPT-4o Vision API, parse response."""
    image_data, media_type = encode_image(image_path)
    data_url = f"data:{media_type};base64,{image_data}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": DETAILED_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content
    return _result_from_raw(raw_text)


def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: str, data: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def _checkpoint_entry_to_result(entry: dict) -> GPTResult:
    return GPTResult(
        front_wall=entry.get("front_wall", entry.get("material", "Unknown")),
        sec_front_wall=entry.get("sec_front_wall", "") or "",
        proportion=entry.get("proportion", "1") or "1",
        back_wall=entry.get("back_wall", "") or "",
        wwr=float(entry.get("wwr", -1.0)),
        raw_response=entry.get("raw_response", ""),
    )


def _result_to_checkpoint_entry(result: GPTResult) -> dict:
    return {
        "front_wall": result.front_wall,
        "sec_front_wall": result.sec_front_wall,
        "proportion": result.proportion,
        "back_wall": result.back_wall,
        "wwr": result.wwr,
        "raw_response": result.raw_response,
    }


def run_gpt_batch(
    image_records: list,
    api_key: str,
    model: str,
    requests_per_minute: int = 60,
    checkpoint_dir: str = "results",
    resume: bool = False,
) -> dict[str, GPTResult]:
    """Process all images through GPT-4o Vision with rate limiting and checkpointing."""
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_data = load_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(checkpoint_data.keys())

    client = openai.OpenAI(api_key=api_key)
    interval = 60.0 / requests_per_minute

    results = {}
    for image_id, data in checkpoint_data.items():
        results[image_id] = _checkpoint_entry_to_result(data)

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    for i, record in enumerate(tqdm(remaining, desc="GPT-4o inference")):
        start_time = time.time()
        try:
            result = query_gpt_single(client, record.image_path, model)
        except Exception as e:
            logger.error("GPT failed for %s: %s", record.image_id, e)
            result = GPTResult(
                front_wall="Unknown", sec_front_wall="", proportion="1",
                back_wall="", wwr=-1.0, raw_response=str(e),
            )

        results[record.image_id] = result
        checkpoint_data[record.image_id] = _result_to_checkpoint_entry(result)

        if (i + 1) % 50 == 0:
            save_checkpoint(checkpoint_path, checkpoint_data)
            logger.info("Checkpoint saved at %d images", len(checkpoint_data))

        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

    save_checkpoint(checkpoint_path, checkpoint_data)
    logger.info("GPT-4o: processed %d images", len(results))
    return results
