# 📊 Customer Churn Prediction

> 🚀 End-to-end Machine Learning project to predict customer churn and help businesses reduce revenue loss.

---

## 🔍 Business Problem

Customer churn is one of the biggest challenges in telecom industries.
This project aims to **identify customers likely to leave**, enabling companies to take proactive retention actions.

---

## 📁 Dataset

* **Source:** Telco Customer Churn Dataset
* **Size:** 7,032 rows × 21 features
* **Target Variable:** `Churn` (Yes/No)

---

## ⚙️ Tech Stack

* Python (Pandas, NumPy)
* Data Visualization (Matplotlib, Seaborn)
* Machine Learning (Scikit-learn)
* Model Persistence (Joblib)

---

## 🧠 Models & Performance

| Model                            | Accuracy | Churn Recall |
| -------------------------------- | -------- | ------------ |
| Logistic Regression (Basic)      | 79%      | 52%          |
| Logistic Regression (Balanced) ⭐ | 73%      | **79%**      |
| Random Forest (Tuned)            | 74%      | 73%          |

> ⭐ **Best Model:** Logistic Regression with class balancing (optimized for churn detection)

---

## 📊 Key Insights

* Class imbalance significantly affects churn prediction
* Balanced Logistic Regression improved recall from **52% → 79%**
* Recall is prioritized over accuracy to capture maximum churn cases

---

## 🔄 ML Workflow

1. Data Cleaning (handled missing & incorrect values)
2. Feature Engineering (encoding categorical variables)
3. Train-Test Split (80/20)
4. Model Training & Evaluation
5. Confusion Matrix Analysis
6. Model Saving (`churn_model.pkl`)

---

## 💾 How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 👨‍💻 Author

**Gopal**
🔗 GitHub: https://github.com/GOPAL-D1x1T

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Deploy model using Streamlit / Flask
* Build interactive dashboard
* Experiment with XGBoost / LightGBM
