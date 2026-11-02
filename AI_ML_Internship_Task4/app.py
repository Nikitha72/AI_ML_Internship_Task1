import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# =====================================================
# 1. APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter student information to receive an estimated "
    "pass/fail prediction."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student-mat.csv"
MODEL_FILE = BASE_DIR / "best_student_model.joblib"


# =====================================================
# 2. LOAD DATASET
# =====================================================

def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"student-mat.csv not found in: {BASE_DIR}"
        )

    # Your current CSV uses commas as separators
    df = pd.read_csv(DATA_FILE, sep=",")

    # Clean column names
    df.columns = df.columns.str.strip()

    if "G3" not in df.columns:
        raise ValueError(
            "Dataset must contain the G3 final grade column."
        )

    # Convert final grade into numeric values
    df["G3"] = pd.to_numeric(
        df["G3"],
        errors="coerce"
    )

    df = df.drop_duplicates()
    df = df.dropna(subset=["G3"]).copy()

    if df.empty:
        raise ValueError("Dataset contains no valid G3 records.")

    return df


# =====================================================
# 3. TRAIN AND SAVE MODEL
# =====================================================

def train_and_save_model():

    df = load_dataset()

    # Target: 1 = Pass, 0 = Fail
    df["Pass"] = (df["G3"] >= 10).astype(int)

    # Exclude G1, G2 and G3 to avoid data leakage
    X = df.drop(
        columns=["G1", "G2", "G3", "Pass"],
        errors="ignore"
    )

    y = df["Pass"]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain both Pass and Fail records."
        )

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Machine learning models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    best_model = None
    best_model_name = None
    best_f1 = -1

    results = []

    # Train and evaluate models
    for name, estimator in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        score = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": score
        })

        if score > best_f1:
            best_f1 = score
            best_model = pipeline
            best_model_name = name

    # Save the best model
    bundle = {
        "model": best_model,
        "feature_columns": X.columns.tolist(),
        "best_model_name": best_model_name,
        "pass_threshold": 10
    }

    joblib.dump(bundle, MODEL_FILE)

    # Save model comparison
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        BASE_DIR / "model_comparison.csv",
        index=False
    )

    return bundle


# =====================================================
# 4. LOAD OR TRAIN MODEL
# =====================================================

@st.cache_resource
def load_bundle():

    # Check whether an existing model can be reused
    if MODEL_FILE.exists():

        try:
            bundle = joblib.load(MODEL_FILE)

            required_keys = [
                "model",
                "feature_columns",
                "best_model_name"
            ]

            if all(key in bundle for key in required_keys):

                df = load_dataset()

                expected_columns = df.drop(
                    columns=["G1", "G2", "G3"],
                    errors="ignore"
                ).columns.tolist()

                if bundle["feature_columns"] == expected_columns:
                    return bundle

        except Exception:
            pass

    # Train a fresh model if no valid model exists
    return train_and_save_model()


# =====================================================
# 5. DISPLAY MODEL STATUS
# =====================================================

try:

    with st.spinner(
        "Loading or training machine learning models..."
    ):

        bundle = load_bundle()

    model = bundle["model"]

    feature_columns = bundle["feature_columns"]

    st.success(
        "Machine learning model loaded successfully!"
    )

    st.info(
        f"Selected model: {bundle['best_model_name']}"
    )

except Exception as exc:

    st.error(
        f"Unable to load or train model: {exc}"
    )

    st.stop()


# =====================================================
# 6. STUDENT INPUT OPTIONS
# =====================================================

categorical_options = {

    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],

    "Mjob": [
        "teacher", "health", "services",
        "at_home", "other"
    ],

    "Fjob": [
        "teacher", "health", "services",
        "at_home", "other"
    ],

    "reason": [
        "home", "reputation", "course", "other"
    ],

    "guardian": [
        "mother", "father", "other"
    ],

    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"]
}


numeric_defaults = {

    "age": (15, 22, 17),
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1),
    "studytime": (1, 4, 2),
    "failures": (0, 4, 0),
    "famrel": (1, 5, 4),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "Dalc": (1, 5, 1),
    "Walc": (1, 5, 1),
    "health": (1, 5, 3),
    "absences": (0, 30, 2)

}


# =====================================================
# 7. STUDENT INPUT FORM
# =====================================================

st.subheader("📝 Enter Student Details")

input_data = {}

with st.form("student_form"):

    for col in feature_columns:

        if col in categorical_options:

            input_data[col] = st.selectbox(
                col,
                categorical_options[col]
            )

        elif col in numeric_defaults:

            low, high, default = numeric_defaults[col]

            input_data[col] = st.number_input(
                col,
                min_value=low,
                max_value=high,
                value=default,
                step=1
            )

        else:

            st.error(
                f"Input control not configured for: {col}"
            )

    submitted = st.form_submit_button(
        "🎯 Predict Student Result"
    )


# =====================================================
# 8. MAKE PREDICTION
# =====================================================

if submitted:

    try:

        row = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        prediction = int(
            model.predict(row)[0]
        )

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.success(
                "🎉 Estimated Result: PASS"
            )

        else:

            st.warning(
                "Estimated Result: FAIL"
            )

        # Display prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(row)[0]

            classes = list(model.classes_)

            if 1 in classes:

                pass_index = classes.index(1)

                pass_probability = probabilities[pass_index]

                st.metric(
                    "Estimated Pass Probability",
                    f"{pass_probability * 100:.2f}%"
                )

        st.caption(
            "This prediction is an estimate based on the "
            "training dataset, not a guarantee of actual "
            "academic performance."
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )