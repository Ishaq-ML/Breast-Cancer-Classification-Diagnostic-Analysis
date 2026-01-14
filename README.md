# Breast Cancer Classification Analysis

## Overview
This project applies supervised machine learning techniques to diagnose breast cancer tumors. By analyzing features computed from digitized images of cell nuclei (such as radius, texture, and smoothness), the model predicts whether a tumor is **Malignant (1)** or **Benign (0)**.

The project demonstrates an end-to-end data science workflow, including data visualization, manual feature engineering, hyperparameter tuning via GridSearch, and comprehensive model evaluation.

## Dataset
The analysis uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
- **Entries:** 569 instances
- **Features:** 30 numeric attributes (radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Diagnosis (M = Malignant, B = Benign)

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:**
  - `pandas` & `numpy` (Data Manipulation)
  - `matplotlib` & `seaborn` (Visualization)
  - `scikit-learn` (Modeling, Preprocessing, Evaluation)

## Methodology

### 1. Data Cleaning & EDA
- Removed unnecessary columns (`id`, `Unnamed: 32`).
- Verified data integrity (no missing values).
- visualized feature distributions using histograms to distinguish between Malignant and Benign characteristics.
- Checked class balance (Benign: 357, Malignant: 212).

### 2. Preprocessing
- **Label Encoding:** Converted diagnosis labels (`M` -> 1, `B` -> 0).
- **Train/Test Split:** 80% Training, 20% Testing (`random_state=44`).
- **Feature Selection:** Manually dropped features with lower predictive power or high noise, specifically: `texture_se`, `smoothness_se`, and `fractal_dimension_mean`.
- **Scaling:** Applied `StandardScaler` to normalize feature variance.

### 3. Modeling
Used `scikit-learn` Pipelines to prevent data leakage during cross-validation.
- **Logistic Regression:** Tuned regularization parameter `C`.
- **Support Vector Classifier (SVC):** Used Linear kernel, tuned parameter `C`.

### 4. Hyperparameter Tuning
Implemented `GridSearchCV` to find the optimal parameters:
- **Logistic Regression Best Params:** `{'C': 0.1}`
- **SVC Best Params:** `{'C': 0.1}`

## Results & Performance

Both models performed exceptionally well on the test set.

### ROC / AUC
- **Logistic Regression:** AUC = 0.999
- **SVC:** AUC = 0.999

### Final Evaluation (Logistic Regression)
The final model was evaluated on the test set with the following confusion matrix results:
- **True Negatives:** 75
- **False Positives:** 0
- **False Negatives:** 1
- **True Positives:** 38

**Classification Report:**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99% |
| **Precision** | 1.00 |
| **Recall** | 0.97 |
| **F1-Score** | 0.99 |

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
