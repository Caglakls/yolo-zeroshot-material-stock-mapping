import json
import logging
import os
import re
from dataclasses import dataclass

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class LLaVAResult:
    material: str
    wwr: float
    raw_response: str


LLAVA_PROMPT = (
    "USER: <image>\n"
    "Analyze this building facade image. Provide:\n"
    "1. The primary wall cladding material. Choose one: Brick, Stucco, Vinyl, "
    "Decorative stone, Aluminum Composite, Metal, Fibercement.\n"
    "2. The estimated Window-to-Wall Ratio (WWR) as a decimal between 0.0 and 1.0.\n\n"
    'Respond in JSON format only: {"material": "<material>", "wwr": <number>}\n'
    "ASSISTANT:"
)

# Keywords for fallback material extraction
MATERIAL_KEYWORDS = {
    "brick": "Brick",
    "stucco": "Stucco",
    "plaster": "Stucco",
    "vinyl": "Vinyl",
    "stone": "Decorative stone",
    "aluminum": "Aluminum Composite",
    "composite": "Aluminum Composite",
    "metal": "Metal",
    "steel": "Metal",
    "fibercement": "Fibercement",
    "fiber cement": "Fibercement",
    "cement board": "Fibercement",
    "hardie": "Fibercement",
}


def load_llava_model(model_name: str, device: str = "cuda"):
    """Load LLaVA-1.5-7B with 4-bit quantization."""
    logger.info("Loading LLaVA model: %s (4-bit quantized)", model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    logger.info("LLaVA model loaded successfully")
    return model, processor


def parse_llava_response(raw: str) -> tuple[str, float]:
    """Parse LLaVA's text response to extract material and WWR."""
    # Try JSON parsing
    try:
        match = re.search(r"\{[^}]+\}", raw)
        if match:
            parsed = json.loads(match.group())
            material = str(parsed.get("material", "Unknown"))
            wwr = float(parsed.get("wwr", -1.0))
            if 0.0 <= wwr <= 1.0:
                return material, wwr
            return material, -1.0
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback for structured output
    mat_match = re.search(r'"material"\s*:\s*"([^"]+)"', raw)
    wwr_match = re.search(r'"wwr"\s*:\s*([\d.]+)', raw)

    if mat_match:
        material = mat_match.group(1)
    else:
        # Keyword scan fallback
        lower = raw.lower()
        material = "Unknown"
        for keyword, label in MATERIAL_KEYWORDS.items():
            if keyword in lower:
                material = label
                break

    wwr = -1.0
    if wwr_match:
        try:
            val = float(wwr_match.group(1))
            if 0.0 <= val <= 1.0:
                wwr = val
        except ValueError:
            pass

    if wwr < 0:
        # Try to find any decimal between 0 and 1 in the text
        decimals = re.findall(r"0\.\d+", raw)
        for d in decimals:
            val = float(d)
            if 0.0 < val < 1.0:
                wwr = val
                break

    return material, wwr


def run_llava_single(
    model,
    processor,
    image_path: str,
    device: str,
) -> LLaVAResult:
    """Process one image through LLaVA."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=LLAVA_PROMPT, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    # Decode only the generated tokens (skip the input)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    raw_text = processor.decode(generated, skip_special_tokens=True).strip()

    material, wwr = parse_llava_response(raw_text)
    return LLaVAResult(material=material, wwr=wwr, raw_response=raw_text)


def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: str, data: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def run_llava_batch(
    image_records: list,
    model_name: str,
    device: str = "cuda",
    checkpoint_dir: str = "results",
    resume: bool = False,
) -> dict[str, LLaVAResult]:
    """Process all images sequentially through LLaVA."""
    checkpoint_path = os.path.join(checkpoint_dir, "llava_checkpoint.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_data = load_checkpoint(checkpoint_path) if resume else {}
    completed_ids = set(checkpoint_data.keys())

    model, processor = load_llava_model(model_name, device)

    results = {}
    for image_id, data in checkpoint_data.items():
        results[image_id] = LLaVAResult(
            material=data["material"],
            wwr=data["wwr"],
            raw_response=data["raw_response"],
        )

    remaining = [r for r in image_records if r.image_id not in completed_ids]
    if completed_ids:
        logger.info("Resuming: %d already done, %d remaining", len(completed_ids), len(remaining))

    for i, record in enumerate(tqdm(remaining, desc="LLaVA inference")):
        try:
            result = run_llava_single(model, processor, record.image_path, device)
            results[record.image_id] = result
            checkpoint_data[record.image_id] = {
                "material": result.material,
                "wwr": result.wwr,
                "raw_response": result.raw_response,
            }
        except Exception as e:
            logger.error("LLaVA failed for %s: %s", record.image_id, e)
            results[record.image_id] = LLaVAResult(
                material="Unknown", wwr=-1.0, raw_response=str(e)
            )
            checkpoint_data[record.image_id] = {
                "material": "Unknown",
                "wwr": -1.0,
                "raw_response": str(e),
            }

        if (i + 1) % 50 == 0:
            save_checkpoint(checkpoint_path, checkpoint_data)
            logger.info("Checkpoint saved at %d images", len(checkpoint_data))

    save_checkpoint(checkpoint_path, checkpoint_data)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    logger.info("LLaVA: processed %d images", len(results))
    return results
