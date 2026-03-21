import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import shap

# Loading the data for model training and model building
df = pd.read_csv(r'D:\e2eML\telco_churn_azure_ml\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Cleaning the data after the initial analysis of our EDA

df = df.drop(columns  = ['customerID',
                         'TotalCharges',
                         'PhoneService'])

# Defining the target variable and the features for model training and model building
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

# Adding numericla nd categorical columsn to perform the ncessary training and model building
num_cols = []
cat_cols = []

for _ in df.columns:
    if df[_].nunique() > 5:
        num_cols.append(_)
    else:
        cat_cols.append(_)
        
# Remoing the churn column from the categorical columns
cat_cols.remove('Churn')

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Model
# model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model = DecisionTreeClassifier(random_state=42)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', model)
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.columns)

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

# 2. Print Classification Report
# This gives you Precision, Recall, and F1-Score per class
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 3. Create and Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_)

# Plotting with a nice colour map
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix')
plt.show()


auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", auc)

# Save model
joblib.dump(pipeline, "model.pkl")


# 1. Create the explainer (use the model inside your pipeline)
# If using a pipeline, ensure the model is the final step
model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(model)

# 2. Calculate SHAP values for your test set
# Note: If you used a ColumnTransformer, you must transform X_test first!
X_test_transformed = pipeline.named_steps['prep'].transform(X_test)
feature_names = pipeline.named_steps['prep'].get_feature_names_out()
shap_values = explainer(X_test_transformed)
shap_values.feature_names = list(feature_names)

# 3. Visualise the first prediction (index 0)
shap.plots.waterfall(shap_values[0][:, 1])

shap.plots.beeswarm(shap_values[:, :, 1])
