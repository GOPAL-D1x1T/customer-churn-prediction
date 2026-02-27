
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 1️⃣ Load Dataset
# ===============================
data = pd.read_csv("../data/churn_data.csv")

print("Dataset Loaded")
print("Shape:", data.shape)

# ===============================
# 2️⃣ Basic Cleaning
# ===============================

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Drop CustomerID
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Drop missing values
data = data.dropna()

# ===============================
# 3️⃣ Feature & Target Split
# ===============================

X = data.drop("Churn", axis=1)
y = data["Churn"].map({"No": 0, "Yes": 1})

# ===============================
# 4️⃣ Identify Column Types
# ===============================

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# ===============================
# 5️⃣ Preprocessing Pipeline
# ===============================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ===============================
# 6️⃣ Model Pipeline
# ===============================

model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# ===============================
# 7️⃣ Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 8️⃣ Train Model
# ===============================

model_pipeline.fit(X_train, y_train)

print("Model Training Completed")

# ===============================
# 9️⃣ Prediction & Evaluation
# ===============================

predictions = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ===============================
# 🔟 Save Model
# ===============================

joblib.dump(model_pipeline, "../model.pkl")
print("\nModel saved as model.pkl")

# ===============================
# 📊 Confusion Matrix Visualization
# ===============================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
