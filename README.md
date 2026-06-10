# yolo-zeroshot-material-stock-mapping
A computer vision workflow for mapping building-level façade material mixtures and window-to-wall ratio from Google Street View imagery using supervised YOLO and zero-shot vision-language models.

This repository contains code, model outputs, and documentation for a workflow that maps building-level façade material mixtures and window-to-wall ratio from Google Street View imagery using supervised YOLO instance segmentation and zero-shot vision-language models.

# GSV Façade Material Mapping

This repository contains the code, model outputs, documentation, and interactive maps developed for the study:

**To Train or Not to Train: Supervised and Zero-Shot Façade Material Mapping for Urban Building Stocks**

The project develops a reproducible workflow for mapping building-level façade material mixtures and window-to-wall ratio (WWR) from Google Street View (GSV) imagery using two computer vision approaches: a supervised YOLOv8 instance segmentation model and zero-shot vision-language models.

The workflow was applied to Fishtown, Philadelphia, a historic rowhouse neighborhood, to generate GIS-ready building-level façade material and WWR data for future material stock, demolition, retrofit, and circular economy analysis.

---

## Repository Overview

This repository includes:

* Final YOLO model weights and processed YOLO outputs
* Zero-shot inference specifications, prompts, and processed VLM outputs
* Curated and raw GSV building-level datasets with dominant material, material fractions, and WWR
* YOLO–zero-shot agreement and uncertainty outputs
* Static figures and interactive maps
* Documentation for data structure, annotation, model outputs, and map interpretation

---

## Research Workflow

The workflow includes the following major steps:

1. Collect building footprint data for the study area.
2. Retrieve Google Street View images for target buildings.
3. Create a curated façade image dataset.
4. Manually annotate façade materials and windows.
5. Train a supervised YOLOv8 instance segmentation model.
6. Run zero-shot vision-language models on curated and raw GSV images.
7. Calculate dominant material, material fractions, and WWR.
8. Join predictions to building footprints.
9. Generate building-level maps.
10. Compare YOLO and zero-shot predictions to create an agreement and uncertainty layer.

---

## Data

The study uses three main data sources:

1. **Building footprints**
   Building footprint polygons and attributes from Philadelphia open data.

2. **Curated GSV façade dataset**
   A manually curated and annotated dataset of façade images used for YOLO training, validation, testing, and zero-shot evaluation.

3. **Raw GSV streetscape dataset**
   A larger neighborhood-scale GSV dataset used for model deployment and building-level mapping.

### Important Note on Google Street View Images

Due to Google Street View licensing and API terms, this repository does not redistribute the full raw GSV image dataset. Instead, the repository provides image metadata, processed prediction outputs, annotations where permitted, trained model weights, code, figures, and interactive maps. Users with appropriate Google Street View API access can regenerate image inputs using the provided metadata and scripts.

---

## Supervised YOLO Pipeline

The supervised pipeline uses a YOLOv8 instance segmentation model trained on manually annotated façade images.

The model detects eight classes:

* Aluminum composite material panels / ACM
* Brick
* Decorative stone
* Fiber cement
* Metal
* Stucco
* Vinyl
* Window

The YOLO training and deployment process produced the following outputs:

1. The final trained model weights, which allow the model to be reloaded for inference without retraining.
2. A complete building-level dataset for each curated image, including model-predicted dominant material and per-class material ratios with corrected mismatches and annotation-based WWR.
3. A raw GSV prediction dataset containing material ratios, WWR estimates, dominant material labels, and detection status for approximately 5,000 raw GSV buildings.
4. An annotated curated dataset of legacy rowhouses to be used in future studies.
5. Interactive maps showing material fractions, dominant material, and WWR for each building.

Main YOLO outputs include:

```text
outputs/yolo/
├── curated_wwr_from_annotations.csv
├── curated_material_ratios.csv
├── curated_full_results.csv
└── raw_gsv_results.csv
```

The trained YOLO model weights are located in:

```text
models/best.pt
```

---

## Zero-Shot Pipeline

The zero-shot workflow uses vision-language models to estimate façade material information and WWR without task-specific training or fine-tuning.

The final zero-shot comparison includes:

* Claude-Sonnet-4
* Gemini-3-Flash
* GPT-4o

Unlike the supervised YOLO pipeline, the zero-shot workflow does not produce trained model weights because no task-specific training is performed. Instead, reproducibility is supported through model-specific inference scripts, structured prompts, parsing logic, material normalization utilities, and saved prediction outputs.

The zero-shot inference and deployment process produced the following outputs:

1. Reusable model-specific inference scripts for Claude-Sonnet-4, Gemini-3-Flash, and GPT-4o.
2. Structured prompt, material normalization, parsing, and WWR utility scripts.
3. Prediction files for the curated benchmark dataset, including dominant material, secondary material, material proportions, and WWR estimates for each model.
4. Raw GSV prediction files for the neighborhood-scale deployment dataset.
5. Mismatch files documenting cases where zero-shot predictions differed from manually derived ground-truth labels.
6. Summary statistics and overview figures reporting model-level performance.
7. Interactive maps showing dominant material, material fractions, WWR, and model agreement.

Main zero-shot outputs include:

```text
outputs/zero_shot/
├── curated_full_results_claude.csv
├── curated_full_results_gemini.csv
├── curated_full_results_gpt.csv
├── raw_gsv_results_claude.csv
├── raw_gsv_results_gemini.csv
├── raw_gsv_results_gpt.csv
├── mismatches.csv
└── summary_stats.txt
```

Zero-shot model scripts include:

```text
src/zero_shot/
├── claude_vision.py
├── gemini_vision.py
├── gpt_vision.py
├── prompts.py
├── wwr_utils.py
└── materials_utils.py
```

Additional model scripts tested during development may also be included, such as CLIP, LLaVA, Grounding DINO, Grounding DINO + SAM, and YOLO-World.

---

## Interactive Maps

Interactive building-level maps are provided in the `interactive_maps/` folder. These maps show:

* Dominant façade material
* Material fractions by class
* Window-to-wall ratio
* YOLO and zero-shot comparison
* Model agreement and uncertainty

Example files:

```text
interactive_maps/
├── yolo_dominant_material.html
├── zeroshot_dominant_material.html
├── yolo_material_fractions.html
├── zeroshot_material_fractions.html
├── yolo_wwr.html
├── zeroshot_wwr.html
└── model_agreement.html
```

---

## Model Agreement and Uncertainty

YOLO and zero-shot predictions are compared at the building level using dominant material agreement and material-mixture similarity.

Agreement categories are defined as:

| Category           | Rule                                             | Interpretation                                          |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------- |
| High agreement     | Same dominant material and low L1 distance       | Same dominant material and similar proportions          |
| Moderate agreement | Same dominant material but moderate L1 distance  | Same dominant material but different proportions        |
| Moderate agreement | Different dominant material but low L1 distance  | Dominant material differs, but mixtures are still close |
| Low agreement      | Different dominant material and high L1 distance | Dominant material and proportions differ                |
| Low agreement      | Strong vector-level disagreement                 | Large difference in material mixture vectors            |

The agreement map is intended as a confidence layer, not as ground-truth validation. Low-agreement buildings indicate where manual review, additional labeling, or field verification may be needed.

---

## Main Outputs

The repository provides:

* YOLO model weights
* Curated benchmark prediction files
* Raw GSV deployment prediction files
* Zero-shot prediction files for Claude, Gemini, and GPT-4o
* WWR estimates
* Dominant material labels
* Per-class material fractions
* Model agreement and uncertainty outputs
* Static figures
* Interactive maps

---

## Limitations

This workflow has several limitations:

* GSV images may be outdated, occluded, blurry, or poorly aligned.
* Some images do not clearly show the target building.
* Raw streetscape images often contain multiple attached buildings.
* Building isolation may introduce uncertainty.
* The mapped material data represent only the visible street-facing façade.
* WWR from street-level imagery is approximate.
* Minority material classes are more difficult to detect due to limited training examples and visual ambiguity.
* Model agreement is a confidence indicator, not ground-truth validation.

---

## Citation

If you use this repository, please cite the associated paper:

```text
Keles, C., Shen, Y., Un, C., Fetter-Garcia, E., Craig, M., Nally, K., Liu, F., & Cruz Rios, F. 
To Train or Not to Train: Supervised and Zero-Shot Façade Material Mapping for Urban Building Stocks.
[Journal information / DOI to be added]
```

A `CITATION.cff` file is included for citation metadata.

---

## Data Availability

The code, trained YOLO model weights, processed façade material and WWR outputs, annotation metadata, performance results, and interactive maps are available in this repository.

Due to Google Street View licensing restrictions, full raw GSV images are not redistributed. Image retrieval metadata and derived model outputs are provided to support reproducibility.

---

## License

The code is released under the MIT License unless otherwise noted. Derived datasets and documentation are released under CC BY 4.0 unless otherwise noted. Google Street View imagery is not redistributed and remains subject to Google Maps Platform terms.

---

## Contact

For questions about this repository, please contact:

**Cagla Keles**
PhD Candidate, Architectural Engineering
Drexel University
[ck976@drexel.edu]
