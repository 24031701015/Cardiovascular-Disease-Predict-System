import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Cardiovascular Disease Prediction", page_icon="❤️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("cardio_train.csv", sep=";")

@st.cache_resource
def train_models():
    df = load_data().drop_duplicates().copy()

    q1, q3 = df["ap_hi"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[df["ap_hi"].between(lo, hi)].copy()

    X = df.drop(columns="cardio")
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, X.columns.tolist(), X_test, y_test, df

models, feature_names, X_test, y_test, cleaned_df = train_models()

st.title("❤️ Cardiovascular Disease Prediction")
st.write("Machine-learning prediction app based on the Cardiovascular Disease dataset.")

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Select a model", list(models.keys()))

st.subheader("Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (days)", min_value=10000, max_value=30000, value=18000)
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)

with col2:
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {
        1: "Normal", 2: "Above Normal", 3: "Well Above Normal"
    }[x])
    gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {
        1: "Normal", 2: "Above Normal", 3: "Well Above Normal"
    }[x])

with col3:
    smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    alco = st.selectbox("Alcohol intake", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    active = st.selectbox("Physically active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    patient_id = st.number_input("Patient ID", min_value=1, value=1)

input_data = pd.DataFrame([{
    "id": patient_id,
    "age": age,
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
}])[feature_names]

if st.button("Predict Cardiovascular Disease", type="primary"):
    model = models[model_name]
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        st.error(f"Prediction: Cardiovascular disease detected (class 1)")
    else:
        st.success(f"Prediction: No cardiovascular disease (class 0)")

    st.metric("Estimated disease probability", f"{probability:.2%}")

st.divider()

st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Cleaned records", f"{len(cleaned_df):,}")
c2.metric("Features", len(feature_names))
c3.metric("Selected model", model_name)

with st.expander("Preview cleaned dataset"):
    st.dataframe(cleaned_df.head(20), use_container_width=True)

st.caption("For educational/project demonstration only. This prediction should not be used as medical advice.")
