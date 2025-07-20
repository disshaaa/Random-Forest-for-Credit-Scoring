# 💳 Random Forest for Credit Scoring

This project predicts the **credit risk** (good or bad) of individuals using financial and demographic data. It uses the **German Credit Dataset** and implements a complete machine learning pipeline using **Scikit-learn** and **Colab**.

---

## 📊 Dataset

**Source:** [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)  
- 1000 samples  
- 20 features (mixed: categorical + numerical)  
- Target: `CreditRisk` (1 = Good Credit, 0 = Bad Credit)

---

## 🧭 Workflow Overview

### ✅ 1. Data Loading & Exploration
- Loaded data from UCI and assigned proper column names.
- Performed exploratory data analysis using `ydata-profiling`.
- Checked for missing values, data types, and class distribution.

### 📈 2. Visualization
- Visualized feature distributions and target imbalance.
- Noted imbalance: 70% good credit, 30% bad credit.

### ⚙️ 3. Preprocessing
- Encoded categorical columns using `LabelEncoder`.
- Scaled numerical features with `StandardScaler`.

### ⚖️ 4. Class Imbalance Handling
- Used **SMOTE** to balance training data.
- Final training set: 560 good, 560 bad credit samples.

### 🌲 5. Model Training
- Trained a **Random Forest Classifier** on the balanced training set.

### 🔍 6. Model Evaluation
- Metrics used:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
  - ROC Curve + AUC Score

### 🔧 7. Hyperparameter Tuning
- Applied `RandomizedSearchCV` with 10 iterations and 5-fold cross-validation.
- Tuned parameters:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 15, 20]
  - `min_samples_split`: [5, 10]
  - `min_samples_leaf`: [2, 4]
  - `max_features`: ['sqrt']
  - `bootstrap`: [True]
- Overfitting controlled: Train Accuracy 0.96 → Test Accuracy 0.735

### 🔁 8. Cross-Validation
- 5-fold F1 cross-validation on balanced train set:
  - **Average F1 Score:** 0.847

### 📊 9. Feature Importance
- Plotted top 10 important features using `feature_importances_`.

## 💾 Model Saving & Loading

```python
import joblib
# Save
joblib.dump(best_rf, "tuned_random_forest_model.joblib")

# Load
model = joblib.load("tuned_random_forest_model.joblib")
```
---
---

# 💽 Streamlit Web App

A user-friendly Streamlit app is included in app.py. It allows users to input all 20 features using dropdowns (for categorical) and sliders/inputs (for numerical), and instantly receive predictions.

### 📂 Files Used
- app.py: Streamlit app interface
- tuned_random_forest_model.joblib: Final trained model
- label_encoders.pkl: LabelEncoders for each categorical feature
- scaler.pkl: Scaler used for numeric standardization
- mappings.json: Human-readable mapping for dropdowns
- columns.json: Feature type classifications

### 🚀 How to Run Streamlit App
```
streamlit run app.py
```
Use the dropdowns to fill all features → click Predict to see whether the applicant is creditworthy or not.
---

## 📈 Final Results Summary

| Metric              | Score    |
|---------------------|----------|
| Train Accuracy      | 0.963    |
| Test Accuracy       | 0.735    |
| AUC Score           | 0.788    |
| CV Avg F1 Score     | 0.847    |

---

## 🛠 Tech Stack

- Python (Colab)
- scikit-learn
- imbalanced-learn
- matplotlib & seaborn
- pandas
- ydata-profiling
- Streamlit (for app deployment)

---
