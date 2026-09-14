#  Random Forest Models for Studying Nanocarrier Targeting to Tumor Tissues

This repository contains the full data preprocessing and machine learning pipeline developed for the Nano-Tumor Data Analysis project (MBA Thesis, 2023).

---

## Summary

Nanocarriers (NCs) have significantly improved the delivery and accumulation of chemotherapeutic agents in tumor tissues. However, unsatisfactory physicochemical and biological properties often result in low Delivery Efficiency (DE). Understanding the exact relationship between these properties and delivery performance is crucial for advancing nanomedicine.

This study evaluated whether machine learning models—specifically **Random Forest (RF)**—could reliably predict the delivery efficiency of nanocarriers to tumor tissues using physicochemical properties and therapeutic strategies as predictor variables. 

Using data extracted from the **Nano-Tumor Database**, the analysis revealed that literature-mined data presented high heterogeneity and weak feature relevance, ultimately refuting the initial hypothesis. This project underscores the critical importance of evaluating data quality, recognizing the limitations of literature-mined databases, and reporting unbiased scientific findings in data science workflows.

---

## Objectives & Hypotheses

* **Primary Objective:** Build and evaluate predictive Random Forest models to estimate Nanocarrier Delivery Efficiency (DE) at different time points (24 h, 168 h, and maximum DE).
* **Predictor Variables:** Physicochemical properties of NCs and therapeutic strategies.
* **Target Variables:** Delivery Efficiency (DE) at 24 h, 168 h, and $DE_{max}$.
* **Hypothesis Evaluation:** Assess whether literature-aggregated database features hold sufficient predictive power for target site delivery.

---

## Methodology & Modeling Workflow

1. **Dataset:** Extracted from the **Nano-Tumor Database**.
2. **Preprocessing & Feature Selection:** Cleaned and structured data features for model consumption.
3. **Model Construction:** Built using the `randomForest` package in **R**.
4. **Performance Evaluation:** Evaluated predicted vs. observed DE across three target metrics ($DE_{24h}$, $DE_{168h}$, $DE_{max}$) using:
   * **$R^2$** (Coefficient of Determination)
   * **MAE** (Mean Absolute Error)
   * **RMSE** (Root Mean Squared Error)

---

## Tech Stack & Dependencies

The project is implemented in **R** using the following key packages:

* **Core Data Manipulation & Visualization:** `tidyverse` (`dplyr`, `ggplot2`, `magrittr`), `readxl`
* **Machine Learning & Modeling:** `caret`, `h2o`, `randomForest`, `gbm`, `glmnet`, `e1071`, `kernlab`, `LiblineaR`, `Boruta`
* **Multivariate Analysis & Clustering:** `FactoMineR`, `factoextra`
* **Parallel Processing:** `doParallel`
