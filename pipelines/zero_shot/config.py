import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(PROJECT_DIR, "curated images_merged")
CSV_PATH = os.path.join(PROJECT_DIR, "curated dataset", "Clean_Image_IDs_merged.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Claude API (Anthropic). Set ANTHROPIC_API_KEY in the environment.
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_REQUESTS_PER_MINUTE = 500  # latency-bound (~3.7s/req), no artificial limit

# OpenAI API (GPT-4o)
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "",
)
GPT_MODEL = "gpt-4o"
GPT_REQUESTS_PER_MINUTE = 500  # latency-bound (~1.2s/req), no artificial limit

# Google AI Studio (Gemini)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_REQUESTS_PER_MINUTE = 500  # latency-bound (~10s/req), no artificial limit

# Local models -- material classification
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

# WWR detection models
YOLO_WORLD_MODEL_SIZE = "s"
YOLO_WORLD_CONF_THRESHOLD = 0.15

GDINO_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.20

SAM_MODEL_NAME = "facebook/sam-vit-base"

# Display names for reports and CSV headers (exact model IDs)
CLAUDE_DISPLAY = "Claude-Sonnet-4"
GPT_DISPLAY = "GPT-4o"
GEMINI_DISPLAY = "Gemini-3-Flash"
CLIP_DISPLAY = "CLIP-ViT-L14"
GDINO_SAM_DISPLAY = "GDINO-Tiny+SAM-Base"
GDINO_DISPLAY = "GDINO-Tiny"
YOLO_WORLD_DISPLAY = "YOLO-World-S"

# 7 canonical material labels matching ground truth
MATERIAL_LABELS = [
    "Brick",
    "Stucco",
    "Vinyl",
    "Decorative stone",
    "Aluminum Composite",
    "Metal",
    "Fibercement",
]

# Map common model outputs to canonical labels
MATERIAL_ALIASES = {
    "brick": "Brick",
    "red brick": "Brick",
    "clay brick": "Brick",
    "masonry": "Brick",
    "brick masonry": "Brick",
    "exposed brick": "Brick",
    "stucco": "Stucco",
    "plaster": "Stucco",
    "rendered": "Stucco",
    "render": "Stucco",
    "cement render": "Stucco",
    "vinyl": "Vinyl",
    "vinyl siding": "Vinyl",
    "plastic siding": "Vinyl",
    "decorative stone": "Decorative stone",
    "stone": "Decorative stone",
    "stone veneer": "Decorative stone",
    "natural stone": "Decorative stone",
    "limestone": "Decorative stone",
    "granite": "Decorative stone",
    "marble": "Decorative stone",
    "aluminum composite": "Aluminum Composite",
    "aluminum": "Aluminum Composite",
    "aluminium": "Aluminum Composite",
    "acm": "Aluminum Composite",
    "aluminum composite panel": "Aluminum Composite",
    "aluminium composite": "Aluminum Composite",
    "aluminum panel": "Aluminum Composite",
    "composite panel": "Aluminum Composite",
    "metal": "Metal",
    "metal panel": "Metal",
    "metal siding": "Metal",
    "corrugated metal": "Metal",
    "steel": "Metal",
    "steel panel": "Metal",
    "metal cladding": "Metal",
    "fibercement": "Fibercement",
    "fiber cement": "Fibercement",
    "fibre cement": "Fibercement",
    "cement board": "Fibercement",
    "hardie board": "Fibercement",
    "fiber cement board": "Fibercement",
    "fibre cement board": "Fibercement",
    "fiber cement siding": "Fibercement",
    "cementitious": "Fibercement",
}

# Filename corrections for known CSV-to-file mismatches
FILENAME_CORRECTIONS = {
    "CMX_439_e_gir": "CMX_439_e_girard",
    "CMX_441_e_gir": "CMX_441_e_girard",
    "RS_1323_e_montgomery": "RS_1323_montgomery",
    "RS_2202_e_sus": "RS_2202_04_e_sus",
}
