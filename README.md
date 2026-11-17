# Beyond Accuracy: Diagnosing Systematic Biases in Classical ML Weather Models Using SHAP Explainability

# Dataset now indexed at OpenAIRE 
[![Indexed in OpenAIRE](https://img.shields.io/badge/Indexed%20in-OpenAIRE-00549F?logo=openaire&logoColor=white)](https://zenodo.org/records/21313125)


This repository contains the reproducible research workflow for the IEEE COMPAS 2026 submission titled "Beyond Accuracy: Diagnosing Systematic Biases in Classical ML Weather Models Using SHAP Explainability". It supports the full experimental pipeline for preprocessing weather data, prototyping classical machine learning models, and producing SHAP-based explainability analyses for the manuscript.

## Project purpose

The repository is organized to support three core stages of the workflow:

1. Data preparation and reshaping
2. Model prototyping and evaluation
3. Explainability analysis with SHAP

The structure is designed to make experiments easy to reproduce and to keep the analysis pipeline modular.

## Repository structure

The repository currently contains the following top-level items:

- [data/](data/)
  Stores input data and processed outputs. This folder is intended for the raw source file and the derived CSVs produced by preprocessing.

- [notebooks/](notebooks/)
  Contains the interactive Jupyter notebooks used for data exploration, model prototyping, and SHAP diagnostics.

- [src/](src/)
  Holds the reusable Python pipeline modules for preprocessing, model training, evaluation utilities, and explainability analysis.

- [figures/](figures/)
  Stores generated plots and visual outputs for exploratory analysis and manuscript-ready figures.

- [paper/](paper/)
  Contains manuscript-related materials such as the abstract draft and IEEE template resources.

- [tests/](tests/)
  Includes regression tests for the preprocessing and pipeline components.

- [logs/](logs/)
  Reserved for runtime or experiment logs.

- [models/](models/)
  Stores trained models and any related artifact files.

- [config.yaml](config.yaml)
  Project configuration file used by the workflow.

- [Execution.txt](Execution.txt)
  Contains execution notes or run metadata.

- [ARCHITECTURE.md](ARCHITECTURE.md)
  Documents the intended repository layout.

- [README.md](README.md)
  Project overview and usage guide.

- [requirements.txt](requirements.txt)
  Python dependency list.

- [.gitignore](.gitignore)
  Git ignore rules for large or local-only files.

## Data Availability

The raw and processed dataset used in this study is publicly available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21313125.svg)](https://doi.org/10.5281/zenodo.21313125)

- **Source:** NASA POWER (Prediction Of Worldwide Energy Resources)
- **Location:** Dhaka, Bangladesh (23.8103°N, 90.4125°E)
- **Parameters:** PRECTOTCORR (corrected precipitation), T2M (temperature at 2 meters)
- **Temporal coverage:** 1981–2025 (daily resolution)
- **License:** CC-BY 4.0

**Dataset citation (plain text):**
```
Nasim, MD Mahfuz Karim. (2026). Dataset for: Beyond Accuracy — Diagnosing Systematic Biases 
in Classical ML Weather Models Using SHAP Explainability [Data set]. Zenodo. 
https://doi.org/10.5281/zenodo.21313125
```

**Dataset citation (BibTeX):**
```bibtex
@dataset{nasim2026dataset,
  author    = {Nasim, MD Mahfuz Karim},
  title     = {Dataset for: Beyond Accuracy --- Diagnosing Systematic Biases in Classical ML Weather Models Using SHAP Explainability},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21313125},
  url       = {https://doi.org/10.5281/zenodo.21313125}
}
```

## Main workflow

The expected workflow is:

1. Place the raw input data in data/raw.
2. Run the preprocessing pipeline to produce a processed dataset in data/processed.
3. Use the notebooks for exploration and prototyping.
4. Train baseline models and save artifacts.
5. Generate SHAP diagnostics and export figures.

## Key files

- [ARCHITECTURE.md](ARCHITECTURE.md)
  Describes the intended project folder structure.

- [requirements.txt](requirements.txt)
  Lists the main Python dependencies required to run the notebooks and scripts.

- [src/data_preprocessing.py](src/data_preprocessing.py)
  Contains data cleaning and preprocessing helpers.

- [src/model_training.py](src/model_training.py)
  Contains baseline model definitions and training utilities.

- [src/shap_diagnostics.py](src/shap_diagnostics.py)
  Implements SHAP-based explainability workflows.

- [src/utils.py](src/utils.py)
  Provides reusable evaluation and utility functions.

- [notebooks/data_exploration.ipynb](notebooks/data_exploration.ipynb)
  Notebook for inspecting the project and data.

- [notebooks/model_prototyping.ipynb](notebooks/model_prototyping.ipynb)
  Notebook for prototyping models.

- [notebooks/shap_diagnostics.ipynb](notebooks/shap_diagnostics.ipynb)
  Notebook for SHAP explainability experiments.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

## Notes

- The raw data files are not meant to be modified directly.
- Processed outputs should be written to the processed data folder.
- Figures and trained model artifacts should be stored under the appropriate output directories for reproducibility.

## Summary

This repository is not just a generic analysis starter; it is a structured research codebase designed to support a publishable IEEE paper. It moves from raw weather data to reproducible preprocessing, baseline modeling, and SHAP-based interpretability, producing the kinds of artifacts needed for a formal scientific submission: transparent experiments, modular code, documented workflows, and manuscript-ready figures. The project is organized so that each stage of the pipeline can be audited, extended, and ultimately included in a polished IEEE conference paper.

## Citing this work

A formal citation with full venue and page details will be added once the COMPAS 2026 proceedings are published. In the meantime, please use the following provisional citations:

**Plain text:**
```
Nasim, MD Mahfuz Karim. (2026). Beyond Accuracy: Diagnosing Systematic Biases in Classical 
ML Weather Models Using SHAP Explainability. IEEE COMPAS 2026 (in press).
```

**BibTeX:**
```bibtex
@inproceedings{nasim2026beyondaccuracy,
  author    = {Nasim, MD Mahfuz Karim},
  title     = {Beyond Accuracy: Diagnosing Systematic Biases in Classical ML Weather Models Using SHAP Explainability},
  booktitle = {Proceedings of the IEEE COMPAS 2026},
  year      = {2026},
  note      = {In press}
}
```

If you use the dataset itself, please cite it separately using the Zenodo DOI listed in the Data Availability section above.
