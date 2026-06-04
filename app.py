
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    model        = joblib.load("diabetes_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
> Predict the likelihood of **diabetes** based on a patient's
> medical history and demographic information.
>
> *For educational purposes only — not a substitute for clinical diagnosis.*
""")
st.divider()

st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Enter the patient's details below:")

def get_user_input():
    age             = st.sidebar.slider("Age (years)", 0, 80, 45)
    gender          = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
    bmi             = st.sidebar.number_input("BMI", 10.0, 95.0, 27.0, step=0.5)
    hba1c           = st.sidebar.number_input("HbA1c Level (%)", 3.5, 9.0, 5.5, step=0.1)
    blood_glucose   = st.sidebar.number_input("Blood Glucose Level (mg/dL)", 80, 300, 120)
    hypertension    = st.sidebar.selectbox("Hypertension?", ["No", "Yes"])
    heart_disease   = st.sidebar.selectbox("Heart Disease?", ["No", "Yes"])
    smoking_history = st.sidebar.selectbox("Smoking History",
                                           ["never", "former", "current",
                                            "not current", "ever", "No Info"])
    race = st.sidebar.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian",
                                         "Hispanic", "Other"])
    race_aa = 1 if race == "AfricanAmerican" else 0
    race_as = 1 if race == "Asian"           else 0
    race_ca = 1 if race == "Caucasian"       else 0
    race_hi = 1 if race == "Hispanic"        else 0
    race_ot = 1 if race == "Other"           else 0

    gender_map  = {"Female": 0, "Male": 1, "Other": 2}
    smoke_map   = {"No Info": 0, "current": 1, "ever": 2,
                   "former": 3, "never": 4, "not current": 5}
    hypert_val  = 1 if hypertension == "Yes" else 0
    heart_val   = 1 if heart_disease == "Yes" else 0

    age_group_val = (0 if age <= 18 else 1 if age <= 35 else
                     2 if age <= 50 else 3 if age <= 65 else 4)
    bmi_cat_val   = (0 if bmi < 18.5 else 1 if bmi < 25 else
                     2 if bmi < 30 else 3)
    risk_score    = round(
        (hba1c / 9.0) * 0.35 +
        (blood_glucose / 300) * 0.35 +
        (bmi / 95.0) * 0.15 +
        (age / 80.0) * 0.10 +
        hypert_val * 0.03 +
        heart_val  * 0.02, 4)
    high_hba1c   = 1 if hba1c >= 6.5 else 0
    high_glucose = 1 if blood_glucose >= 200 else 0
    has_comorbid = 1 if (hypert_val == 1 and heart_val == 1) else 0
    is_senior    = 1 if age >= 60 else 0

    data = {
        "gender"             : gender_map.get(gender, 0),
        "age"                : age,
        "race:AfricanAmerican": race_aa,
        "race:Asian"         : race_as,
        "race:Caucasian"     : race_ca,
        "race:Hispanic"      : race_hi,
        "race:Other"         : race_ot,
        "hypertension"       : hypert_val,
        "heart_disease"      : heart_val,
        "smoking_history"    : smoke_map.get(smoking_history, 0),
        "bmi"                : bmi,
        "hbA1c_level"        : hba1c,
        "blood_glucose_level": blood_glucose,
        "age_group"          : age_group_val,
        "bmi_category"       : bmi_cat_val,
        "risk_score"         : risk_score,
        "high_hba1c"         : high_hba1c,
        "high_glucose"       : high_glucose,
        "has_comorbidity"    : has_comorbid,
        "is_senior"          : is_senior,
    }
    return pd.DataFrame([data])

input_df = get_user_input()

for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_cols]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("🔮 Prediction Result")
    try:
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        no_diab_pct = probability[0] * 100
        diab_pct    = probability[1] * 100

        if prediction == 0:
            st.success("✅ Prediction: **No Diabetes Detected**")
        else:
            st.error("⚠️ Prediction: **Diabetes Risk Detected**")

        st.markdown("#### Confidence Scores")
        col_a, col_b = st.columns(2)
        col_a.metric("No Diabetes", f"{no_diab_pct:.1f}%")
        col_b.metric("Diabetic",    f"{diab_pct:.1f}%")

        if diab_pct < 20:
            risk = "🟢 Low Risk"
        elif diab_pct < 50:
            risk = "🟡 Moderate Risk"
        else:
            risk = "🔴 High Risk"

        st.markdown(f"#### Risk Level: {risk}")
        st.progress(int(diab_pct))

        st.markdown("#### Key Risk Indicators")
        st.write(f"- **HbA1c ≥ 6.5%**: {'⚠️ Yes' if input_df['high_hba1c'].values[0] == 1 else '✅ No'}")
        st.write(f"- **Blood Glucose ≥ 200 mg/dL**: {'⚠️ Yes' if input_df['high_glucose'].values[0] == 1 else '✅ No'}")
        st.write(f"- **Has Comorbidity**: {'⚠️ Yes' if input_df['has_comorbidity'].values[0] == 1 else '✅ No'}")
        st.write(f"- **Senior (≥60)**: {'⚠️ Yes' if input_df['is_senior'].values[0] == 1 else '✅ No'}")

    except Exception as e:
        st.warning(f"Prediction error: {e}")
        st.info("Ensure diabetes_model.pkl, scaler.pkl, feature_cols.pkl are in the same folder.")

st.divider()
st.markdown("""
**Dataset:** Diabetes Prediction — Kaggle | 100,000 rows × 16 columns
**Model:** Tuned via GridSearchCV with 5-fold stratified cross-validation
**Metrics:** Precision, Recall, F1-Score, AUC-ROC
*For educational use only.*
""")
