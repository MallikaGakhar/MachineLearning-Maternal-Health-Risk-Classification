# Risk Level Classification Using kNN, Decision Trees, and SVM

## Project Overview

This project involves developing classification models to predict risk levels (high, medium, low) using machine learning techniques on a structured dataset. The dataset contains encoded and normalized features suitable for distance-based algorithms and classification tasks.

The main objectives are:

- To explore and justify the choice of machine learning models.
- To train, tune, and evaluate multiple models (k-Nearest Neighbors, Decision Trees, and Support Vector Machines).
- To compare model performances using accuracy, precision, recall, and F1-score.
- To improve model robustness using ensemble methods such as bagging.
- To provide insights into model stability using holdout and k-fold cross-validation.

---

## Models Used and Rationale

### 1. k-Nearest Neighbors (kNN)

- Works well with normalized continuous features.
- Non-parametric and adaptable without assuming data distribution.
- Effective in finding patterns based on distance computations.

### 2. Decision Trees

- Flexible, non-parametric, and suitable for categorical classification.
- Can model complex feature-target relationships and non-linear boundaries.
- Useful when interpretability is important.

### 3. Support Vector Machines (SVM)

- Strong classifier, especially with non-linear kernels (e.g., RBF).
- Performs well with normalized continuous features.
- Good at reducing false positives (high precision).

### Models Not Used

- Linear and multiple regression: inappropriate due to categorical target variable.
- Regularization regression methods (Lasso, Ridge): designed for regression, not classification.
- Neural Networks: dataset too small (~1,014 rows) for effective generalization, simpler models suffice.

---

## Dataset

- The dataset consists of encoded and normalized features.
- Target variable: RiskLevel, categorized as high, medium, or low.
- Dataset size: ~1,014 rows.

---

## Model Training and Evaluation

### Data Splitting

- Training/Test split: 90% training, 10% testing (holdout method).
- Features and labels separated and preprocessed accordingly.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

### Holdout Evaluation

- kNN accuracy on holdout: ~67.65%
- High precision (~0.94), balanced recall (~0.76), and F1-score (~0.84).
- Indicates low false positives and moderate sensitivity.

### k-Fold Cross-Validation (k=10)

- Average accuracy: ~59%
- Highlights variability across folds, suggesting model sensitivity to data splits.
- Holdout provides a sufficient estimate given dataset size.

---

## Hyperparameter Tuning

### kNN

- Tuned number of neighbors (k) from 1 to 15 (odd values).
- Optimal k = 1, achieving highest accuracy (~88.23%).

### Decision Tree

- Tuned max depth (5, 10, 15) and min split (10, 20).
- Best accuracy: ~66%, similar to original model performance.

### Support Vector Machine

- Discussed hyperparameters to tune (kernel, C, gamma).
- Suggested for further work.

---

## Model Comparison Summary

| Model          | Accuracy (Original) | Precision | Recall | F1-Score |
|----------------|---------------------|-----------|--------|----------|
| kNN            | ~71%                | High      | Moderate| ~82%     |
| Decision Tree  | ~67%                | Moderate  | High   | ~75%     |
| SVM            | ~71%                | Highest   | Moderate| ~81%     |

- kNN and SVM show similar overall accuracy and F1-scores.
- SVM excels in precision (reducing false positives).
- Decision Tree has higher recall (better at identifying positives).

---

## Ensemble Model: Bagging with Decision Trees

- Applied bootstrap aggregating (bagging) using `randomForest`.
- Bagging improved accuracy to ~85%, reducing overfitting and variance.
- Demonstrates benefit of ensemble methods for stability and performance.

---

## Usage

### Prerequisites

- R environment with the following packages:
  - `class` (for kNN)
  - `rpart` (for Decision Trees)
  - `randomForest` (for bagging)
  - `caret` (for cross-validation folds)
  - Other standard data manipulation packages (`dplyr`, `tidyr`, etc.)

### Running the Project

1. Load and preprocess the dataset (encoding, normalization).
2. Split data into training and testing sets.
3. Train kNN, Decision Tree, and SVM models.
4. Evaluate models using holdout and k-fold cross-validation.
5. Tune hyperparameters as shown in the scripts.
6. Apply bagging for decision trees to improve model performance.
7. Compare results and interpret metrics.

---

## Conclusion

This project highlights the process of selecting appropriate machine learning models for a multiclass classification task, tuning and evaluating them, and applying ensemble methods to improve stability and accuracy. The results indicate that kNN and SVM provide the best trade-off between precision and recall, while bagging notably enhances decision tree performance.

---
