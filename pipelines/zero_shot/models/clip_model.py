import logging
from dataclasses import dataclass

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


@dataclass
class CLIPResult:
    material: str
    scores: dict[str, float]


# Ensemble text prompts per material class
MATERIAL_PROMPTS = {
    "Brick": [
        "a building facade made of brick",
        "a brick wall on a building",
        "red brick building exterior",
        "masonry brick facade",
    ],
    "Stucco": [
        "a building facade made of stucco",
        "a stucco plastered wall on a building",
        "smooth plastered building exterior",
        "rendered stucco facade",
    ],
    "Vinyl": [
        "a building facade with vinyl siding",
        "vinyl siding on a building",
        "plastic vinyl exterior cladding",
        "horizontal vinyl siding facade",
    ],
    "Decorative stone": [
        "a building facade made of decorative stone",
        "stone veneer on a building wall",
        "decorative stone cladding facade",
        "natural stone building exterior",
    ],
    "Aluminum Composite": [
        "a building facade with aluminum composite panels",
        "aluminum composite material cladding on a building",
        "metal composite panel cladding",
        "modern aluminum panel facade",
    ],
    "Metal": [
        "a building facade made of metal panels",
        "corrugated metal wall on a building",
        "metal siding building exterior",
        "steel panel facade",
    ],
    "Fibercement": [
        "a building facade with fiber cement panels",
        "fiber cement board siding on a building",
        "cement board cladding facade",
        "hardie board building exterior",
    ],
}


def load_clip_model(model_name: str, device: str = "cuda"):
    """Load CLIP model and processor."""
    logger.info("Loading CLIP model: %s", model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _to_tensor(features) -> torch.Tensor:
    """Extract tensor from model output (handles both raw tensors and dataclass outputs)."""
    if isinstance(features, torch.Tensor):
        return features
    # transformers 5.x may return a dataclass with .pooler_output or similar
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0, :]
    raise TypeError(f"Unexpected features type: {type(features)}")


def precompute_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: dict[str, list[str]],
    device: str,
) -> dict[str, torch.Tensor]:
    """Precompute averaged text embeddings for each material."""
    embeddings = {}
    with torch.no_grad():
        for material, texts in prompts.items():
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
            text_features = _to_tensor(model.get_text_features(**inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            avg_embedding = text_features.mean(dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm()
            embeddings[material] = avg_embedding
    return embeddings


def classify_single(
    model: CLIPModel,
    processor: CLIPProcessor,
    text_embeddings: dict[str, torch.Tensor],
    image_path: str,
    device: str,
) -> CLIPResult:
    """Classify one image. Returns material with highest cosine similarity."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = _to_tensor(model.get_image_features(**inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    scores = {}
    for material, text_emb in text_embeddings.items():
        similarity = (image_features @ text_emb.unsqueeze(-1)).squeeze().item()
        scores[material] = similarity

    best_material = max(scores, key=scores.get)
    return CLIPResult(material=best_material, scores=scores)


def run_clip_batch(
    image_records: list,
    model_name: str,
    device: str = "cuda",
) -> dict[str, CLIPResult]:
    """Process all images through CLIP zero-shot classification."""
    model, processor = load_clip_model(model_name, device)
    text_embeddings = precompute_text_embeddings(model, processor, MATERIAL_PROMPTS, device)

    results = {}
    for record in tqdm(image_records, desc="CLIP inference"):
        try:
            result = classify_single(model, processor, text_embeddings, record.image_path, device)
            results[record.image_id] = result
        except Exception as e:
            logger.error("CLIP failed for %s: %s", record.image_id, e)
            results[record.image_id] = CLIPResult(material="Unknown", scores={})

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    logger.info("CLIP: processed %d images", len(results))
    return results
