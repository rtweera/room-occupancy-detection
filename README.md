# ML Notebooks

This repository contains a machine learning notebook that builds an occupancy detection model from environmental sensor data.

## Repository Contents

- `occupancy-model-ML-module-project.ipynb`  
  End-to-end notebook covering EDA, feature engineering, model training, tuning, evaluation, and explainability.

## Project Goal

Predict building/room occupancy (`Occupancy`: 0 or 1) using sensor readings:

- Temperature
- Humidity
- Light
- CO2
- HumidityRatio

## Dataset

The notebook uses three occupancy dataset files:

- `datatraining.txt`
- `datatest.txt`
- `datatest2.txt`

The original notebook expects these files in a Kaggle-style path:

`/kaggle/input/occupancy/`

For local execution, update paths in the data-loading cell accordingly.

## Notebook Workflow

1. **Data loading and merge**
   - Read and combine the three dataset files.
2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis, pair plots, correlation checks, and outlier inspection.
3. **Preprocessing**
   - Datetime parsing and time-order sorting.
   - Time-series-aware cross-validation with `TimeSeriesSplit`.
   - CO2 transformation with Box-Cox.
   - Light discretization with `KBinsDiscretizer`.
4. **Feature engineering**
   - Time-based features (`hour`, `day_of_week`, cyclical encodings).
   - Delta/rate features for selected sensors.
5. **Class imbalance handling**
   - `SMOTE` applied on training folds.
6. **Model comparison**
   - `RandomForestClassifier`
   - `XGBClassifier`
   - `LGBMClassifier`
7. **Pipeline and hyperparameter tuning**
   - Pipeline-based training with `GridSearchCV`.
8. **Evaluation and interpretation**
   - Accuracy, precision/recall/F1, ROC-AUC, confusion matrix, timing.
   - SHAP-based model explainability.

## Main Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `imbalanced-learn`
- `lightgbm`
- `xgboost`
- `shap`

## Environment

- Notebook metadata indicates Python **3.10.14**.

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn lightgbm xgboost shap`
3. Ensure dataset files are available and paths in the notebook are correct.
4. Start Jupyter and run:
   - `occupancy-model-ML-module-project.ipynb`
5. Execute cells in order from top to bottom.

## Outputs You Can Expect

- Fold-wise model performance summaries
- Confusion matrices and classification reports
- Selected/best hyperparameters from tuning
- Inference/training time comparisons
- SHAP plots for feature importance and interpretability

## Notes and Caveats

- This is a notebook-first repository (no standalone Python package/modules).
- The workflow is time-series-aware and avoids random split leakage by using `TimeSeriesSplit`.
- Performance values are generated at runtime in notebook outputs and may vary by environment/version.

