# YOLO and Zero-Shot Façade Material Stock Mapping

This repository contains code, annotation files, processed outputs, figures, maps, and documentation for a computer vision workflow that maps building-level façade material mixtures and window-to-wall ratio (WWR) from Google Street View imagery.

The project compares two approaches for façade material mapping:

1. A supervised YOLOv8 instance segmentation workflow trained on manually annotated façade images.
2. A zero-shot vision-language model workflow using prompted inference without task-specific training.

The workflow was applied to Fishtown, Philadelphia, a historic rowhouse neighborhood, to generate GIS-ready building-level façade material and WWR datasets for future material stock, demolition, retrofit, and circular economy analysis.

## Study

This repository supports the study:

**To Train or Not to Train: Supervised and Zero-Shot Façade Material Mapping for Urban Building Stocks**

The manuscript is currently under review. A formal citation and DOI will be added after publication. 
If you use the code, processed outputs, annotation files, maps, or workflow from this repository before the paper is published, please cite this GitHub repository:

```text
Keles, C. (2026). YOLO and Zero-Shot Façade Material Stock Mapping. GitHub repository. https://github.com/Caglakls/yolo-zeroshot-material-stock-mapping
```

## Repository Contents

This repository includes:

* YOLO annotation files in COCO format for training, validation, and testing
* Processed YOLO prediction outputs for curated and raw GSV datasets
* Zero-shot inference code, prompt/model configuration, and processed outputs
* YOLO and zero-shot confusion matrices and evaluation results
* Building-level agreement layers comparing YOLO and zero-shot predictions
* Static maps and figures
* Interactive HTML maps for YOLO material fractions and WWR
* Documentation for model outputs, data structure, and workflow interpretation

The full raw Google Street View image dataset and trained YOLO model weights are not stored directly in this repository because of file size and licensing considerations.

## Repository Structure

```text
yolo-zeroshot-material-stock-mapping/
│
├── agreement_layer/
│   ├── agreement_layer_claude.csv
│   ├── agreement_layer_gemini.csv
│   ├── agreement_layer_gpt.csv
│   └── build_agreement_layer.py
│
├── confusion_matrices/
│   ├── yolo/
│   └── zero_shot/
│
├── data/
│   ├── train/
│   │   └── _annotations.coco.json
│   ├── valid/
│   │   └── _annotations.coco.json
│   ├── test/
│   │   └── _annotations.coco.json
│   ├── README.roboflow.txt
│   └── benchmark_dataset.csv
│
├── interactive maps/
│   ├── interactive_material fraction map_YOLO.html
│   └── interactive_wwr_map_YOLO.html
│
├── maps/
│   ├── fig1_dominant_material-yolo.png
│   ├── fig1_dominant_material-zeroshot.png
│   ├── fishtown_model_agreement_map.png
│   ├── fishtown_wwr_map_gemini.png
│   ├── fishtown_wwr_map_yolo.png
│   └── additional comparison figures
│
├── outputs/
│   ├── yolo/
│   │   ├── curated_full_results.csv
│   │   ├── curated_material_ratios.csv
│   │   ├── curated_wwr_from_annotations.csv
│   │   ├── raw_gsv_results.csv
│   │   ├── results_overview.png
│   │   ├── weights.png
│   │   └── wwr_comparison.png
│   │
│   └── zero_shot/
│       ├── curated_full_results_claude.csv
│       ├── curated_full_results_gemini.csv
│       ├── curated_full_results_gpt.csv
│       ├── raw_gsv_results_claude.csv
│       ├── raw_gsv_results_gemini.csv
│       ├── raw_gsv_results_gpt.csv
│       ├── per_class_f1_strict_vs_lenient.csv
│       ├── summary_stats.txt
│       └── overview figures
│
├── pipelines/
│   ├── yolo/
│   │   └── Facade_Material_Segmentation_v2 (1).ipynb
│   │
│   └── zero_shot/
│       ├── README.md
│       ├── config.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── main.py
│       ├── materials_utils.py
│       ├── preprocess.py
│       ├── run_new_data.py
│       ├── visualize.py
│       ├── pyproject.toml
│       ├── uv.lock
│       └── models/
│
├── LICENSE
└── README.md
```

## Data

The repository includes the annotation and output files needed to understand and evaluate the workflow.

The `data/` folder contains the COCO-format annotation files for the curated YOLO dataset split into training, validation, and testing subsets. The repository also includes a benchmark dataset file used for evaluation.

The full raw Google Street View image dataset is not redistributed in this repository. Users with appropriate access can regenerate or replace the image inputs using their own Google Street View workflow.

## Supervised YOLO Pipeline

The supervised pipeline uses a YOLOv8 instance segmentation model trained on manually annotated façade images.

The YOLO model was trained to identify the following classes:

* Aluminum composite material panels / ACM
* Brick
* Decorative stone
* Fiber cement
* Metal
* Stucco
* Vinyl
* Window

The repository includes the YOLO training notebook, COCO annotation files, processed prediction outputs, WWR results, and summary figures. The trained YOLO weights are not stored directly in this repository due to file size limitations.

### YOLO outputs

Main YOLO outputs are located in:

```text
outputs/yolo/
```

This folder includes:

* `curated_full_results.csv`
* `curated_material_ratios.csv`
* `curated_wwr_from_annotations.csv`
* `raw_gsv_results.csv`
* `results_overview.png`
* `weights.png`
* `wwr_comparison.png`

### YOLO model weights

The trained YOLO model weights are hosted externally due to file size limitations.

```text
Model weights:
[https://drive.google.com/drive/folders/11x9cH3AhnYFhkVM8dNJg2-swOQVYZWn4?usp=drive_link]
```

The primary model file for reuse is `best.pt`, which can be loaded for inference on new façade or Google Street View images.

## Zero-Shot Pipeline

The zero-shot workflow uses prompted vision-language model inference to estimate façade material composition and WWR without task-specific training or fine-tuning.

The reported zero-shot models include:

* Claude-Sonnet-4
* GPT-4o
* Gemini-3-Flash

Unlike the YOLO workflow, the zero-shot workflow does not produce custom trained weights. Reproducibility is supported through the provided code, model configuration, prompts, material normalization rules, parsing logic, and saved prediction outputs.

Zero-shot code is located in:

```text
pipelines/zero_shot/
```

This folder includes:

* `main.py` for curated dataset evaluation
* `run_new_data.py` for deployment on new images
* `config.py` for model IDs, label space, paths, and API key handling
* `data_loader.py` for input parsing and material normalization
* `evaluation.py` for accuracy, precision, recall, and F1 evaluation
* `materials_utils.py`, `preprocess.py`, and `visualize.py`
* `models/` containing model wrappers and prompt utilities

The zero-shot workflow requires API keys for the selected models. No API keys are stored in this repository.

## Zero-Shot Outputs

Main zero-shot outputs are located in:

```text
outputs/zero_shot/
```

This folder includes curated and raw GSV prediction results for Claude, Gemini, and GPT-4o, as well as summary statistics and overview figures.

Key files include:

* `curated_full_results_claude.csv`
* `curated_full_results_gemini.csv`
* `curated_full_results_gpt.csv`
* `raw_gsv_results_claude.csv`
* `raw_gsv_results_gemini.csv`
* `raw_gsv_results_gpt.csv`
* `per_class_f1_strict_vs_lenient.csv`
* `summary_stats.txt`

## Agreement Layer

YOLO and zero-shot predictions are compared at the building level using dominant material agreement and material-mixture similarity.

Agreement layer files are located in:

```text
agreement_layer/
```

This folder includes agreement outputs for each zero-shot model comparison:

* `agreement_layer_claude.csv`
* `agreement_layer_gemini.csv`
* `agreement_layer_gpt.csv`
* `build_agreement_layer.py`

The agreement layer is intended as a confidence and uncertainty indicator. Low-agreement buildings may require manual review, additional annotation, or field verification.

## Maps and Figures

Static maps and figures are provided in:

```text
maps/
```

These include dominant material maps, WWR maps, model agreement maps, and YOLO–zero-shot comparison figures.

Interactive HTML maps are provided in:

```text
interactive maps/
```

This folder currently includes:

* `interactive_material fraction map_YOLO.html`
* `interactive_wwr_map_YOLO.html`

## Data Availability

The repository provides code, annotation files, processed prediction outputs, evaluation results, agreement layers, static figures, and interactive maps.

The full raw and curated Google Street View image dataset is not redistributed because of file size and Google Street View licensing considerations. The trained YOLO model weights are also hosted externally (see above) because the model file is too large for standard GitHub upload.

External data and model weight links can be added here:

```text
Full image dataset:
[https://drive.google.com/drive/folders/1e4p9jU0IypgQe1OIeGGJdFs1DDYA1aZ8?usp=drive_link]

Curated image dataset:
[https://drive.google.com/drive/folders/1hvAyykOnl941x0wDwrlVJKmdaWcdhXq7?usp=drive_link]
```

## Limitations

This workflow has several limitations:

* Google Street View images may be outdated, blurry, occluded, or poorly aligned.
* Some images may not clearly show the target building.
* Raw streetscape images can contain multiple attached buildings.
* Building isolation and façade visibility introduce uncertainty.
* The mapped material data represent only the visible street-facing façade.
* WWR estimates from street-level imagery are approximate.
* Minority material classes are more difficult to classify because of limited examples and visual similarity.
* The agreement layer is a confidence indicator, not a substitute for ground-truth validation.

## Citation

This repository accompanies a manuscript that is currently under review.

If you use the code, processed outputs, annotation files, maps, or workflow from this repository before the paper is published, please cite this GitHub repository:

```text
Keles, C. (2026). YOLO and Zero-Shot Façade Material Stock Mapping. GitHub repository. https://github.com/Caglakls/yolo-zeroshot-material-stock-mapping
```
Once the associated paper is published, this section will be updated with the full journal citation and DOI.

## License

The code is released under the MIT License unless otherwise noted.

Derived datasets, documentation, figures, and outputs may be used with attribution unless otherwise restricted. Google Street View imagery is not redistributed and remains subject to Google Maps Platform terms.

## Contact

For questions about this repository, please contact:

**Cagla Keles**
PhD Candidate, Architectural Engineering
Drexel University
[ck976@drexel.edu]

