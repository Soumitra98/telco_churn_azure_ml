import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, classification_report, confusion_matrix

# --- 1. Load and Clean Data ---
data_path = r'D:\e2eML\telco_churn_azure_ml\data\WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

# Drop redundant or problematic columns
df = df.drop(columns=['customerID', 'TotalCharges', 'PhoneService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV'])

# Prepare Target and Features
# Note: X does NOT contain 'Churn', so our pipeline won't look for it later
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

# --- 2. Feature Selection Logic ---
# Identifying columns based on cardinality
num_cols = [col for col in X.columns if X[col].nunique() > 5]
cat_cols = [col for col in X.columns if X[col].nunique() <= 5]

# --- 3. Preprocessing & Pipeline Construction ---
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Build the base pipeline
# base_pipeline = Pipeline([
#     ('prep', preprocessor),
#     ('dt', DecisionTreeClassifier(random_state=42))
# ])

base_pipeline = Pipeline(
    [
        ('prep', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ]
)

# --- 4. Hyperparameter Tuning with GridSearchCV ---
# We use 'dt__' prefix to target the DecisionTreeClassifier step
# param_grid = {
#     'dt__max_depth': [3, 5, 10, None],
#     'dt__min_samples_split': [2, 10, 20],
#     'dt__criterion': ['gini', 'entropy']
# }

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__bootstrap': [True, False]
}

# Use StratifiedKFold to handle class imbalance
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',  # F1 is better for churn than accuracy
    n_jobs=-1,
    verbose=1
)

# --- 5. Training ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")

# --- 6. Evaluation ---
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


# --- 8. Save Artifacts ---
joblib.dump(best_pipeline, "telco_churn_model.pkl")