# DiToS: Multi-Tier Data Augmentation and Imbalance Handling for PV Fault Detection

This repository provides the official implementation of **DiToS**, a multi-tier data augmentation and balancing framework for robust photovoltaic (PV) panel fault and dust detection under class imbalance.

DiToS integrates **Stable Diffusion–based synthetic image generation**, **Tomek Link cleaning**, and **SMOTE oversampling**, and evaluates robustness across both binary and multi-class PV fault datasets.

---

## 📌 Overview

Class imbalance is a major challenge in PV fault diagnosis, where faulty or dusty panels are rare compared to clean ones. DiToS addresses this issue through a staged pipeline that:

- Generates realistic minority-class samples using diffusion models  
- Removes ambiguous majority-class samples using Tomek Links  
- Balances the feature space using SMOTE  
- Evaluates robustness using performance, reliability, and statistical metrics  

The framework is designed to be **model-agnostic**, **scalable**, and **deployment-aware**.

---

## 🧠 Key Contributions

- Multi-tier augmentation combining diffusion models with classical imbalance handling techniques  
- Validation on both binary and multi-class PV fault datasets  
- Reliability evaluation using Cohen’s κ and Matthews Correlation Coefficient (MCC)  
- Statistical significance testing using Wilcoxon signed-rank tests  
- Conceptual edge-deployment pipeline for real-time PV monitoring  

---

## 📂 Repository Structure

├── data/
│ ├── raw/ # Original PV images (not included)
│ ├── synthetic/ # Stable Diffusion generated images
│ └── processed/ # Preprocessed datasets
│
├── diffusion/
│ ├── generate_images.py # SDXL image generation script
│ └── prompts.txt # Prompt templates
│
├── models/
│ ├── vit_head.py # ViT-based classifier
│ ├── xgboost_model.py
│ └── svm_model.py
│
├── imbalance/
│ ├── smote.py
│ ├── tomek_link.py
│ └── ditos_pipeline.py
│
├── evaluation/
│ ├── metrics.py # Accuracy, F1, AUC, κ, MCC
│ ├── wilcoxon_test.py
│ └── confusion_matrix.py
│
├── results/
│ ├── tables/
│ └── figures/
│
├── requirements.txt
└── README.md
