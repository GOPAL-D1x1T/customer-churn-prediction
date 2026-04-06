import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\gopal\OneDrive\Desktop\churn_data.csv.csv")
print("Shape:", df.shape)
df.head()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

print("After cleaning shape:", df.shape)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df["Churn"].value_counts())

df.drop(columns=["customerID"], inplace=True)

df = pd.get_dummies(df, drop_first=True)

print("Final shape:", df.shape)

df.isnull().sum()

from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Model with more iterations
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Step 3: Predict
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Scale use
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanced class weight model
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Best Model (Balanced LR)")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=10,          # overfitting 
    min_samples_leaf=5,    # churn class attention
    random_state=42
)
rf_model.fit(X_train, y_train)   # X_train_scaled nahi, direct X_train

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

import joblib

# Best model save 
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved!")
