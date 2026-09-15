# PRCP-1016 — Heart Disease Prediction

## Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year — about 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third occur prematurely in people under 70. Early detection in people at high cardiovascular risk (due to hypertension, diabetes, hyperlipidaemia, or existing disease) can enable earlier management and better outcomes.

This project builds a machine learning model to predict the presence of heart disease in patients using 13 clinical features, and provides data-driven suggestions for hospital screening priorities.

**Domain:** Healthcare

## Tasks

1. **Data Analysis** — Complete exploratory data analysis (EDA) on the dataset.
2. **Model Building** — Train and compare multiple ML classification models to predict heart disease.
3. **Hospital Suggestions** — Recommend how hospitals can use these predictions to act earlier on heart disease risk.

## Dataset

The dataset contains 180 patient records with 14 columns (`patient_id` is a unique random identifier; the remaining 13 are clinical features):

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age in years |
| `sex` | binary | 0 = female, 1 = male |
| `chest_pain_type` | int | Chest pain type (4 values) |
| `resting_blood_pressure` | int | Resting blood pressure |
| `serum_cholesterol_mg_per_dl` | int | Serum cholesterol in mg/dl |
| `fasting_blood_sugar_gt_120_mg_per_dl` | binary | Fasting blood sugar > 120 mg/dl |
| `resting_ekg_results` | int | Resting electrocardiographic results (0, 1, 2) |
| `max_heart_rate_achieved` | int | Maximum heart rate achieved (bpm) |
| `exercise_induced_angina` | binary | Exercise-induced chest pain (0 = False, 1 = True) |
| `oldpeak_eq_st_depression` | float | ST depression induced by exercise relative to rest |
| `slope_of_peak_exercise_st_segment` | int | Slope of the peak exercise ST segment |
| `num_major_vessels` | int | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | categorical | Thallium stress test result: `normal`, `fixed_defect`, `reversible_defect` |
| `heart_disease_present` | binary | **Target** — 0 = no disease, 1 = disease present |

## Project Structure

```
├── heart.ipynb              # Main Jupyter notebook (EDA + modeling, single notebook per project requirements)
├── data/
│   ├── values.csv            # Feature data
│   └── labels.csv            # Target labels
├── description.docx          # Original problem statement document
└── README.md
```

## Approach

### 1. Exploratory Data Analysis
- Checked data quality (no missing values, no duplicates)
- Analyzed target class balance (~56% no disease, 44% disease)
- Visualized distributions of numeric and categorical features
- Detected outliers via IQR method
- Correlation analysis and statistical significance testing (t-test, chi-square) against the target

### 2. Data Preprocessing
- One-hot encoded the categorical `thal` column
- Train/test split (80/20, stratified on target)
- Standardized numeric features with `StandardScaler`

### 3. Model Building & Comparison
Trained and compared 5 classification models:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **0.861** | 0.789 | **0.938** | **0.857** |
| SVM | 0.861 | **0.824** | 0.875 | 0.848 |
| Logistic Regression | 0.833 | 0.778 | 0.875 | 0.824 |
| KNN | 0.833 | 0.778 | 0.875 | 0.824 |
| Decision Tree | 0.778 | 0.750 | 0.750 | 0.750 |

**Selected model: Random Forest** — chosen for production due to its highest recall (critical in healthcare, where missing an actual heart disease case is costlier than a false alarm), plus interpretable feature importances.

### 4. Key Findings — Feature Importance
Top predictors of heart disease presence: `thal` (thallium stress test result), `age`, `max_heart_rate_achieved`, `oldpeak_eq_st_depression` (ST depression), and `serum_cholesterol_mg_per_dl`.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
   ```
3. Place `values.csv` and `labels.csv` in the `data/` folder (or update the path in the notebook)
4. Open and run `heart.ipynb` in Jupyter Notebook / JupyterLab

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Challenges Faced

- **Mixed data types**: `thal` was text-based, requiring one-hot encoding before correlation/model use.
- **Small dataset size** (180 records): addressed via stratified splitting to preserve class balance.
- **Mild class imbalance**: evaluated Precision/Recall/F1 alongside Accuracy rather than relying on Accuracy alone.
- **Feature scale differences**: standardized all features before training distance-based models (KNN, SVM).

## Suggestions to Hospital

- Prioritize thallium stress tests (`thal`) as a key diagnostic signal — the single strongest predictor in this dataset.
- Pay close attention to ST depression (`oldpeak`) and abnormally low max heart rate during exercise stress testing, especially in older patients.
- Use the model as a **screening aid**, not a standalone diagnostic tool — flagged high-risk patients should be prioritized for further cardiologist evaluation.
- Given the small sample size, retrain/validate the model periodically as more patient data becomes available.




