# Customer Churn Prediction — ML Deployment

## Business Problem
A Telco provider needed to identify customers likely to churn 
before they leave, enabling proactive retention strategies.

## Solution
Built an end-to-end ML pipeline from data preprocessing to 
live Flask deployment, predicting churn probability in real time.

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- XGBoost, Random Forest, Logistic Regression
- SMOTE for class imbalance handling
- Flask for web deployment
- Pickle for model serialization

## Model Performance
- Evaluated using AUC-ROC, F1-score, Precision, Recall
- Selected model based on maximising recall (fraud detection priority)

## Features
- 36 engineered features including service attributes and tenure groups
- One-hot encoded categorical variables
- Real-time churn probability scoring via browser

## How to Run
1. Clone the repo
2. pip install -r requirements.txt
3. python app.py
4. Open http://127.0.0.1:5000
