# AF/AFL Ablation Outcome Prediction Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-survival](https://img.shields.io/badge/scikit--survival-latest-green.svg)](https://scikit-survival.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.hlc.2023.12.016-blue)](https://doi.org/10.1016/j.hlc.2023.12.016)

## Overview

This repository contains the implementation of machine learning models for predicting adverse outcomes following catheter ablation treatment for atrial fibrillation (AF) and/or atrial flutter (AFL).

The models were developed using a comprehensive linked dataset from New South Wales, Australia, incorporating hospital administrative data, prescription medicine claims, emergency department presentations, and death registrations.

## Research Paper

This code accompanies the research published in Heart, Lung and Circulation:

**Predicting Adverse Outcomes Following Catheter Ablation Treatment for Atrial Flutter/Fibrillation**

DOI: [https://doi.org/10.1016/j.hlc.2023.12.016](https://doi.org/10.1016/j.hlc.2023.12.016)

## Key Features

- Implementation of traditional and deep survival models
- Models for predicting two distinct outcomes:
  1. Major bleeding events
  2. Composite outcome (heart failure, stroke, cardiac arrest, death)
- Feature importance analysis and visualization
- Evaluation metrics including concordance index

## Dataset

The study cohort included 3,285 patients who received catheter ablation for AF and/or AFL in New South Wales, Australia. Due to privacy regulations, the raw data cannot be shared publicly. However, we provide:

- Data preprocessing scripts
- Feature engineering pipelines
- Synthetic data generators for testing the models

## Models

The repository implements several survival analysis models:

- Cox Proportional Hazards
- Random Survival Forest
- Gradient Boosting Survival Models
- Deep Survival Networks

## Results

Our models achieved:
- Composite outcome prediction: concordance index >0.79
- Major bleeding events prediction: concordance index <0.66

Feature importance analyses identified the following as key predictors:
- Comorbidities indicating poor health
- Older age
- Therapies for heart failure and AF/AFL management

## Feature Importance Analysis

The study utilized SHAP (SHapley Additive exPlanations) values to identify the most important features for predicting adverse outcomes:

![SHAP Feature Importance Analysis](https://ars.els-cdn.com/content/image/1-s2.0-S1443950624000039-gr1_lrg.jpg)
*Figure 1: SHAP summary plot for the composite outcome prediction model. Features are ranked by their impact on model predictions.*

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{QUIROZ2024470,
  title = {Predicting Adverse Outcomes Following Catheter Ablation Treatment for Atrial Flutter/Fibrillation},
  journal = {Heart, Lung and Circulation},
  volume = {33},
  number = {4},
  pages = {470-478},
  year = {2024},
  issn = {1443-9506},
  doi = {https://doi.org/10.1016/j.hlc.2023.12.016},
  url = {https://www.sciencedirect.com/science/article/pii/S1443950624000039},
  author = {Juan C. Quiroz and David Brieger and Louisa R. Jorm and Raymond W. Sy and Benjumin Hsu and Blanca Gallego},
  keywords = {Atrial fibrillation, Catheter ablation, Machine learning, Survival analysis, Treatment outcome},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Juan C. Quiroz
- David Brieger
- Louisa R. Jorm
- Raymond W. Sy
- Benjumin Hsu
- Blanca Gallego

## Contact

For questions about the code or paper, please open an issue in this repository or contact the corresponding author.
