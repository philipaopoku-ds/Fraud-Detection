# 💳 Fraud Detection System

A **Machine Learning-based platform** for detecting fraudulent financial transactions. The project uses Python, scikit-learn, XGBoost, and Random Forest classifiers, and has been deployed as a **Streamlit web application** for interactive use.

---

## 📊 Dataset

- The dataset contains **95,662 transactions** with features such as TransactionId, AccountId, ProductCategory, Amount, TransactionStartTime, and FraudResult (target).
- No missing values and no duplicates.
- Transactions include both **credit and debit**, handled with feature engineering.

---

## ⚙️ Features & Engineering

- **Time-based features:** Extracted Month and Day from `TransactionStartTime`.
- **Transaction type:** Created `Credit_or_Debit` column (0 = credit, 1 = debit).
- **Feature selection:** Dropped redundant columns like TransactionId, CurrencyCode, CountryCode, Amount, and TransactionStartTime.
- **Encoding:** Label encoding for categorical columns (e.g., AccountId, ProductCategory, ProviderId).
- **Scaling:** StandardScaler applied to all features for normalization.

---

## 🧠 Modeling

Three models were trained and evaluated:

1. **Logistic Regression**
   - Accuracy: 0.998
   - Handles imbalanced dataset but lower recall for fraud class.
2. **XGBoost Classifier**
   - Accuracy: 0.996
   - Performance affected by class imbalance.
3. **Random Forest Classifier**
   - Accuracy: 0.999
   - Best performance with F1-score for fraud class: 0.76
   - Confusion matrix visualized with seaborn.

> Note: Dataset is highly imbalanced (fraudulent transactions are rare). Metrics like **recall, precision, and F1-score** are more informative than accuracy.

---

## 📈 Visualizations

- Correlation heatmap of features
- Confusion matrix heatmap for model performance
- Exploratory plots for transaction distribution by Month, Day, and Transaction type

---

## 💻 Deployment

- The project has been **deployed on Streamlit** for real-time fraud detection.
- Users can upload transaction data and get predictions instantly.

**Access the app here:** [Streamlit Fraud Detection App](https://fraud-detection-by-phil.streamlit.app/)

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Streamlit for deployment

---

## 🔗 Project Highlights

- Feature engineering for time and transaction type
- Handling of imbalanced dataset in real-world scenario
- Comparison of Logistic Regression, XGBoost, and Random Forest classifiers
- End-to-end deployment as interactive web app

