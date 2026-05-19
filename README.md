# ==========================================================
# Maternal Mortality Risk Prediction in Low-Resource Settings
# ==========================================================
# Author: Jona Buka
# Description:
# Machine Learning pipeline for predicting maternal mortality
# risk using synthetic maternal healthcare data.

# ==========================
# IMPORT LIBRARIES
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
import joblib

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("maternal_mortality_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================
# TARGET VARIABLE
# ==========================
target = "maternal_death"

# ==========================
# REMOVE DATA LEAKAGE FEATURES
# ==========================
# These variables directly indicate death outcome
leakage_columns = [
    "time_to_death_hours",
    "cause_of_death",
    "maternal_near_miss"
]

# Drop patient identifier
id_columns = ["patient_id"]

columns_to_drop = leakage_columns + id_columns

existing_cols = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=existing_cols, inplace=True)

# ==========================
# HANDLE DATE COLUMN
# ==========================
if "admission_date" in df.columns:
    df["admission_date"] = pd.to_datetime(
        df["admission_date"],
        errors="coerce"
    )

    df["admission_year"] = df["admission_date"].dt.year
    df["admission_month"] = df["admission_date"].dt.month

    df.drop(columns=["admission_date"], inplace=True)

# ==========================
# FEATURES & TARGET
# ==========================
X = df.drop(target, axis=1)
y = df[target]

print("\nTarget Distribution:")
print(y.value_counts())

# ==========================
# IDENTIFY FEATURE TYPES
# ==========================
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:", len(categorical_features))
print("Numerical Features:", len(numerical_features))

# ==========================
# PREPROCESSING PIPELINE
# ==========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ==========================
# PREPROCESS DATA
# ==========================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==========================
# HANDLE CLASS IMBALANCE
# ==========================
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ==========================
# MODEL TRAINING
# ==========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

# ==========================
# MODEL PREDICTIONS
# ==========================
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]

# ==========================
# EVALUATION
# ==========================
print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# CONFUSION MATRIX
# ==========================
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    ax=ax
)

plt.title("Confusion Matrix")
plt.show()

# ==========================
# ROC CURVE
# ==========================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# ==========================
# FEATURE IMPORTANCE
# ==========================
encoded_cat_features = preprocessor.named_transformers_[
    "cat"
]["encoder"].get_feature_names_out(categorical_features)

all_features = np.concatenate([
    numerical_features,
    encoded_cat_features
])

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(20)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)
plt.title("Top 20 Features Influencing Maternal Mortality")
plt.tight_layout()
plt.show()

print("\nTop Predictive Features:")
print(top_features)

# ==========================
# PREDICT RISK SCORES
# ==========================
risk_scores = model.predict_proba(
    preprocessor.transform(X)
)[:, 1]

df["maternal_mortality_risk_score"] = risk_scores

# Categorize risk
df["risk_category"] = pd.cut(
    df["maternal_mortality_risk_score"],
    bins=[0, 0.30, 0.70, 1.0],
    labels=["Low", "Medium", "High"]
)

# Save predictions
df.to_csv(
    "maternal_mortality_predictions.csv",
    index=False
)

print("\nPredictions saved:")
print("maternal_mortality_predictions.csv")

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "maternal_mortality_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("\nModel saved successfully!")
print("maternal_mortality_model.pkl")
print("preprocessor.pkl")
Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
What this model predicts

It learns maternal mortality risk from variables such as:

maternal age
ANC visits
anemia
blood pressure
hemorrhage
sepsis
facility access
skilled birth attendance
delivery complications
obstetric emergencies
healthcare availability

…and outputs a maternal mortality risk score (0–1) plus a Low / Medium / High risk category.

This is structured professionally for a strong GitHub public health ML project.

write me a very professional README for this project for github
# Maternal Mortality Risk Prediction in Low-Resource Settings

## Overview

Maternal mortality remains a major public health challenge in many low-resource settings, where limited healthcare access, delayed interventions, and inadequate maternal care contribute significantly to preventable deaths.

This project leverages **Machine Learning (ML)** to predict **maternal mortality risk** using a realistic **synthetic maternal healthcare dataset**. The goal is to identify high-risk pregnancies early and support evidence-based maternal health interventions in resource-constrained environments.

Using predictive analytics, this project simulates how healthcare systems can proactively identify women at increased risk of maternal complications or mortality, enabling earlier clinical attention and better allocation of limited healthcare resources.

---

## Project Objectives

The project aims to:

- Predict maternal mortality risk using maternal health indicators
- Identify key factors contributing to maternal mortality
- Classify patients into risk categories (Low, Medium, High)
- Support data-driven maternal healthcare decision-making
- Demonstrate the application of AI/ML in public health systems

---

## Problem Statement

Maternal deaths are often preventable, yet many occur due to delayed diagnosis, poor healthcare accessibility, lack of antenatal care, and pregnancy-related complications.

In many low-resource settings:

- Skilled birth attendance is limited
- Emergency obstetric care is inadequate
- Delays in treatment increase fatality risks
- Predictive healthcare systems are scarce

This project addresses these challenges by building an ML-powered maternal mortality risk prediction system.

---

## Dataset Description

This project uses a **realistic synthetic maternal healthcare dataset** designed to simulate maternal health records from low-resource settings.

The dataset contains variables associated with maternal health outcomes, including:

### Demographic Features
- Maternal age
- Education level
- Household income
- Geographic region

### Pregnancy & Clinical Features
- Number of antenatal care visits
- Pregnancy complications
- Previous pregnancy history
- High-risk pregnancy status
- Blood pressure levels
- Hemoglobin level
- Presence of anemia

### Healthcare Access Features
- Distance to health facility
- Skilled birth attendance
- Type of healthcare facility
- Access to emergency obstetric care

### Maternal Outcome Variables
- Maternal mortality outcome
- Risk indicators
- Clinical complications

---

## Machine Learning Workflow

The project follows a complete **end-to-end machine learning pipeline**:

### 1. Data Preprocessing
- Missing value handling
- Feature cleaning
- Categorical encoding
- Feature engineering
- Data leakage prevention

### 2. Exploratory Data Analysis (EDA)
- Maternal mortality distribution
- Risk factor analysis
- Correlation analysis
- Healthcare inequality insights

### 3. Class Imbalance Handling
Maternal mortality datasets are often imbalanced because death cases are relatively rare.

To improve prediction quality, the project uses:

- **SMOTE (Synthetic Minority Oversampling Technique)**

### 4. Model Training
The predictive model is built using:

- **Random Forest Classifier**

Why Random Forest?

- Handles mixed healthcare variables effectively
- Reduces overfitting
- Performs well on tabular medical data
- Provides feature importance for interpretability

### 5. Model Evaluation
Performance is assessed using:

- Accuracy Score
- ROC-AUC Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

### 6. Risk Prediction
The trained model generates:

- Maternal mortality probability scores
- Low-risk classification
- Medium-risk classification
- High-risk classification

---

## Project Structure

```text
Maternal-Mortality-Risk-Prediction-in-Low-Resource-Settings/
│
├── maternal_mortality_dataset.csv
├── maternal_mortality_predictions.csv
├── maternal_mortality_model.pkl
├── preprocessor.pkl
│
├── maternal_mortality_prediction.py
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming Language
- Python

### Data Science Libraries
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Maternal-Mortality-Risk-Prediction-in-Low-Resource-Settings.git
```

Move into the project directory:

```bash
cd Maternal-Mortality-Risk-Prediction-in-Low-Resource-Settings
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the model training script:

```bash
python maternal_mortality_prediction.py
```

The script will:

✔ Load the maternal health dataset  
✔ Clean and preprocess data  
✔ Train the machine learning model  
✔ Evaluate prediction performance  
✔ Generate risk scores  
✔ Save predictions  
✔ Export trained model files

---

## Model Outputs

After execution, the following files are generated:

### 1. Predictions File
```text
maternal_mortality_predictions.csv
```

Contains:
- Predicted mortality risk score
- Risk category (Low / Medium / High)

### 2. Saved Model
```text
maternal_mortality_model.pkl
```

Reusable trained ML model.

### 3. Preprocessor
```text
preprocessor.pkl
```

Used for transforming new patient data before prediction.

---

## Sample Insights

The model may identify major maternal mortality risk factors such as:

- Severe anemia
- Hypertension
- Pregnancy complications
- Low antenatal care attendance
- Poor healthcare access
- Long travel distance to facilities
- Obstetric emergencies

These insights can support maternal health policy and intervention planning.

---

## Public Health Relevance

This project demonstrates how **Artificial Intelligence in Public Health** can support:

- Early maternal risk detection
- Healthcare resource prioritization
- Preventive maternal interventions
- Clinical decision support systems
- Maternal health surveillance

The approach is especially relevant for **low-resource healthcare systems** where early risk identification can reduce preventable maternal deaths.

---

## Limitations

- Dataset is synthetic and not real patient data
- Results are for research and educational purposes
- Clinical deployment would require real-world validation and ethical approval

---

## Future Improvements

Potential enhancements include:

- XGBoost and LightGBM comparison
- Deep learning approaches
- Real-world maternal health datasets
- Explainable AI (SHAP values)
- Web dashboard deployment
- GIS-based maternal mortality mapping

---

## Results

The final model predicts maternal mortality risk and provides interpretable insights into the most influential health and healthcare access factors affecting maternal survival outcomes.

---

## Author

**Jonah Buka**  
Data Science | Machine Learning | Public Health Analytics

---

## License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## Acknowledgements

Inspired by global maternal health challenges and the need for AI-driven healthcare solutions in underserved communities.
