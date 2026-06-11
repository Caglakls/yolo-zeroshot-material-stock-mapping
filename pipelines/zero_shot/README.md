# Zero-shot facade material and WWR pipeline (code)

Training-free pipeline that predicts facade material composition and the
window-to-wall ratio (WWR) for a building image with a single prompted inference
call to a general-purpose vision-language model (VLM). This is the code that
produced output (1) (the inference specification) and generated the curated and
raw-GSV prediction CSVs in the sibling output folders of this package.

## Credentials (no keys are committed)

The three API keys default to empty strings and are read from the environment.
Set the ones you intend to use before running:

```bash
export ANTHROPIC_API_KEY="..."   # Claude
export OPENAI_API_KEY="..."      # GPT-4o
export GOOGLE_API_KEY="..."      # Gemini
# optional: override the Anthropic endpoint (defaults to https://api.anthropic.com)
# export ANTHROPIC_BASE_URL="..."
```

No credentials, endpoints, or private hosts are stored in this code.

## Install and run

```bash
uv sync                                              # install dependencies (pyproject.toml / uv.lock)
uv run python main.py --models claude gpt gemini     # curated dataset, with evaluation
uv run python run_new_data.py --models claude gpt gemini --image-dir <dir>   # deployment on new images (no ground truth)
```

Common flags (both entry points): `--models` (subset of
`claude gpt gemini clip llava yolo_world gdino gdino_sam`, default
`claude gpt gemini`), `--max-images N`, `--resume`, `--device {cuda,cpu}`.

## Reported models

| Display name    | API identifier             |
|-----------------|----------------------------|
| Claude-Sonnet-4 | `claude-sonnet-4-20250514` |
| GPT-4o          | `gpt-4o`                   |
| Gemini-3-Flash  | `gemini-3-flash-preview`   |

The `clip`, `llava`, `yolo_world`, `gdino`, and `gdino_sam` wrappers are included
for completeness (evaluated, not the reported VLMs) and require additional local
model weights and GPU dependencies.

## Output schema and label space

- Prompt and parser: `models/prompts.py`. Each prediction returns the JSON keys
  `front_wall`, `sec_front_wall`, `proportion`, `back_wall`, `wwr`.
- Seven canonical material labels and the output-normalization alias map:
  `config.py` (`MATERIAL_LABELS`, `MATERIAL_ALIASES`): Brick, Stucco, Vinyl,
  Decorative stone, Aluminum Composite, Metal, Fibercement.

## Files

- `main.py` — curated-dataset orchestrator (runs models, then evaluation).
- `run_new_data.py` — deployment runner for new images without ground truth.
- `config.py` — paths, model IDs, label space, alias map (keys read from env).
- `data_loader.py` — input parsing and material normalization.
- `evaluation.py` — accuracy and per-class precision/recall/F1 (lenient:
  primary-or-secondary match; strict primary-only is computed alongside).
- `materials_utils.py`, `preprocess.py`, `visualize.py` — material helpers, GSV
  image preprocessing, per-image visualization.
- `models/` — one wrapper per model, plus `prompts.py` and `wwr_utils.py`.
- `pyproject.toml`, `uv.lock`, `.python-version` — dependency manifest.

## Input data not included

Running the pipeline requires the imagery and label files, which are not part of
this code package: the curated facade images and `curated dataset/
Clean_Image_IDs_merged.csv`, and the raw GSV images for deployment. The
prediction outputs themselves are provided in the sibling output folders
(`2_curated_478_evaluation/`, `3_raw_gsv_4668_deployment/`,
`4_agreement_layer/`).
