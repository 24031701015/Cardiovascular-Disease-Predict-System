import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide"
)



BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cardio_train.csv"

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        st.error(
            f"Dataset not found!\n\n"
            f"Please place 'cardio_train.csv' in:\n"
            f"{BASE_DIR}"
        )
        st.stop()

    df = pd.read_csv(DATA_FILE, sep=";")

    return df


df = load_data()

feature_names = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

@st.cache_resource
def train_model(data):

    X = data[feature_names]
    y = data["cardio"]

    # Use maximum 20,000 records for faster training
    sample_size = min(20000, len(data))

    if len(data) > sample_size:
        sample_data = data.sample(
            n=sample_size,
            random_state=42
        )
    else:
        sample_data = data

    X = sample_data[feature_names]
    y = sample_data["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                solver="liblinear",
                max_iter=300
            )
        )
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, y_test, y_pred


model, accuracy, y_test, y_pred = train_model(df)




st.sidebar.title("❤️ Cardiovascular Prediction")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "Model Performance",
        "Dataset",
        "About Project"
    ]
)




if page == "Prediction":

    st.title("❤️ Cardiovascular Disease Prediction")

    st.write(
        "Enter the patient's information below to predict "
        "the possibility of cardiovascular disease."
    )

    st.divider()


    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        # AGE IN YEARS
        age_years = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=50,
            step=1
        )

        # Convert years to days because dataset uses age in days
        age = age_years * 365

        st.caption(
            f"Model input: approximately {age:,} days"
        )

        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x:
                "Female" if x == 1 else "Male"
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100,
            max_value=220,
            value=165,
            step=1
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=65.0,
            step=0.5
        )

    with col2:

        ap_hi = st.number_input(
            "Systolic Blood Pressure",
            min_value=80,
            max_value=250,
            value=120,
            step=1
        )

        ap_lo = st.number_input(
            "Diastolic Blood Pressure",
            min_value=40,
            max_value=150,
            value=80,
            step=1
        )

        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x:
                {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
        )

        gluc = st.selectbox(
            "Glucose Level",
            options=[1, 2, 3],
            format_func=lambda x:
                {
                    1: "Normal",
                    2: "Above Normal",
                    3: "Well Above Normal"
                }[x]
        )

    st.subheader("🏃 Lifestyle Information")

    col3, col4, col5 = st.columns(3)

    with col3:

        smoke = st.selectbox(
            "Smoking",
            options=[0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    with col4:

        alco = st.selectbox(
            "Alcohol Consumption",
            options=[0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    with col5:

        active = st.selectbox(
            "Physical Activity",
            options=[0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    st.divider()

    if st.button(
        "🔍 Predict Cardiovascular Disease",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                gender,
                height,
                weight,
                ap_hi,
                ap_lo,
                cholesterol,
                gluc,
                smoke,
                alco,
                active
            ]],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        disease_probability = probability[1] * 100

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Cardiovascular disease prediction: POSITIVE"
            )

            st.write(
                f"Estimated probability: "
                f"**{disease_probability:.2f}%**"
            )

        else:

            st.success(
                "✅ Cardiovascular disease prediction: NEGATIVE"
            )

            st.write(
                f"Estimated probability: "
                f"**{disease_probability:.2f}%**"
            )

        st.info(
            "This prediction is generated by a machine-learning "
            "model and should not be considered medical advice."
        )

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "The application uses Logistic Regression with "
        "StandardScaler."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Training Samples",
            f"{min(20000, len(df)):,}"
        )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

elif page == "Dataset":

    st.title("📂 Cardiovascular Dataset")

    st.write(
        f"Dataset contains **{len(df):,} records**."
    )

    st.write(
        f"Number of columns: **{len(df.columns)}**"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )
    st.divider()

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )
elif page == "About Project":

    st.title("ℹ️ About Project")

    st.divider()

  
