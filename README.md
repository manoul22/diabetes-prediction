**Project description:**
This project builds a complete Data Science pipeline — from raw uncleaned data all the way to a deployed web application — to solve a real-world medical problem: predicting diabetes risk from patient data.
The dataset contains 100,000 patient records with 16 features including age, BMI, HbA1c level, blood glucose level, smoking history, hypertension, and heart disease status. Only 8.5% of patients are diabetic, making class imbalance a key challenge.

Diabetes Risk Prediction A machine learning web application that predicts whether a patient has diabetes based on their medical history and demographic data. Project Description This project follows a complete Data Science pipeline:

Data Cleaning — handling hidden null values and duplicates using a composite key strategy Exploratory Data Analysis — 7 visualizations across 5 plot types investigating 6+ variables Feature Engineering — 7 new clinically-motivated features created from existing data Feature Selection — Filter (ANOVA), Embedded (Random Forest), and Wrapper (RFE) methods Modeling — 3 algorithms trained and compared (Logistic Regression, Random Forest, Gradient Boosting) Hyperparameter Tuning — GridSearchCV with 5-fold stratified cross-validation Evaluation — Precision, Recall, F1-Score, AUC-ROC metrics Deployment — Live interactive web app built with Streamlit

Dataset Source: Kaggle — Diabetes Prediction Dataset by iammustafatz 🔗 Download Dataset

you can try this application via:
https://diabetes-prediction-production-08c0.up.railway.app/

you should add your parameters and the program would predict the level of diabetic risk for you.

**Required Libraries**
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn==1.3.2
joblib>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0

**Engineered Features:**
risk_score: Weighted combo: HbA1c (35%) + glucose (35%) + BMI (15%) + age (10%)
high_hba1c: Binary: HbA1c ≥ 6.5% (clinical diabetes threshold)
high_glucose: Binary: blood glucose ≥ 200 mg/dL
has_comorbidity: Binary: has both hypertension AND heart disease
age_group: Age binned into 5 clinical brackets
bmi_category: WHO BMI classification
is_senior: Binary: age ≥ 60

models results are shown in plot 10

Done by: Manoul Mourad Fayez
ID:221001797

**Important note: I had a problem regarding the app(it was working for a week before streamlit changed its sklearn version), and streamlit had blocked my other (original repo) thus, i had to use another account and edit the submission that's why the submission appear to be late. the original repo link:**
https://github.com/ManoulMourad/diabetes-prediction-app
