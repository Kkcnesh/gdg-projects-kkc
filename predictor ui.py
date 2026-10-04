import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Placement Readiness Predictor",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# DATA PREPARATION
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "placement_readiness_synthetic_10000.csv"
    )

    # Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # --------------------------------------
    # Clean backlogs
    # --------------------------------------

    df["backlogs"] = df["backlogs"].fillna("0")

    df["backlogs"] = df["backlogs"].replace({
        "3+": 3
    })

    df["backlogs"] = df["backlogs"].astype(int)

    # --------------------------------------
    # Count programming languages
    # --------------------------------------

    def count_languages(value):

        if pd.isna(value):
            return 0

        value = str(value).strip()

        if value == "":
            return 0

        return len(value.split(";"))

    df["language_count"] = df["languages"].apply(
        count_languages
    )

    # --------------------------------------
    # Count certifications
    # --------------------------------------

    def count_certifications(value):

        if pd.isna(value):
            return 0

        value = str(value).strip()

        if value == "":
            return 0

        return len(value.split(";"))

    df["certification_count"] = df["certifications"].apply(
        count_certifications
    )

    # --------------------------------------
    # Remove irrelevant columns
    # --------------------------------------

    df = df.drop(
        columns=[
            "name",
            "email",
            "other_course",
            "opinion",
            "languages",
            "certifications"
        ]
    )

    return df


# ==========================================
# FEATURE DEFINITIONS
# ==========================================

categorical_cols = [
    "gender",
    "course",
    "branch",
    "year",
    "participation",
    "internship",
    "training"
]

numeric_cols = [
    "age",
    "cgpa",
    "backlogs",
    "projects",
    "tech_skill",
    "comm_skill",
    "apt_skill",
    "team_skill",
    "language_count",
    "certification_count"
]


# ==========================================
# PREPROCESSOR
# ==========================================

def create_preprocessor():

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_cols
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_cols
            )
        ]
    )

    return preprocessor


# ==========================================
# CREATE MODELS
# ==========================================

def create_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                class_weight="balanced"
            )
    }

    return models


# ==========================================
# EVALUATE ALL MODELS
# ==========================================

@st.cache_resource
def evaluate_models():

    df = load_data()

    # Features
    X = df.drop(
        columns=["placement"]
    )

    # Target
    y = df["placement"].map({
        "Not Placed": 0,
        "Placed": 1
    })

    # --------------------------------------
    # Train / Test Split
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models()

    results = []

    for model_name, classifier in models.items():

        # Each model gets its own preprocessing pipeline
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predict test set
        predictions = pipeline.predict(
            X_test
        )

        # Calculate metrics from actual predictions
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
        )

    metrics_df = pd.DataFrame(
        results
    )

    return metrics_df


# ==========================================
# TRAIN FINAL PREDICTION MODEL
# ==========================================

@st.cache_resource
def train_prediction_model():

    df = load_data()

    X = df.drop(
        columns=["placement"]
    )

    y = df["placement"].map({
        "Not Placed": 0,
        "Placed": 1
    })

    prediction_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    # Train final model on all cleaned data
    prediction_model.fit(
        X,
        y
    )

    return prediction_model


# ==========================================
# LOAD MODELS
# ==========================================

metrics_df = evaluate_models()

prediction_model = train_prediction_model()


# ==========================================
# UI
# ==========================================

st.title(
    "🎓 Placement Readiness Predictor"
)

st.write(
    "Enter a student's academic, skill and experience "
    "information to generate a placement prediction."
)

st.divider()


# ==========================================
# STUDENT INFORMATION
# ==========================================

st.subheader(
    "Student Information"
)

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=40,
        value=21
    )

    course = st.selectbox(
        "Course",
        [
            "B.Tech",
            "B.E",
            "BCA",
            "B.Sc",
            "B.Com",
            "M.Tech",
            "MCA",
            "M.Sc",
            "M.Com",
            "Other"
        ]
    )

    branch = st.selectbox(
        "Branch",
        [
            "CSE",
            "IT",
            "ECE",
            "Mechanical",
            "Civil",
            "Electrical"
        ]
    )

    year = st.selectbox(
        "Year",
        [
            "1st Year",
            "2nd Year",
            "3rd Year",
            "4th Year",
            "Completed"
        ]
    )


with col2:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    backlogs = st.selectbox(
        "Backlogs",
        [
            0,
            1,
            2,
            3
        ]
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=50,
        value=3
    )

    tech_skill = st.slider(
        "Technical Skill",
        1,
        5,
        3
    )

    comm_skill = st.slider(
        "Communication Skill",
        1,
        5,
        3
    )

    apt_skill = st.slider(
        "Aptitude Skill",
        1,
        5,
        3
    )

    team_skill = st.slider(
        "Team Skill",
        1,
        5,
        3
    )


# ==========================================
# EXPERIENCE
# ==========================================

st.subheader(
    "Experience & Activities"
)

col1, col2 = st.columns(2)


with col1:

    participation = st.selectbox(
        "Participation",
        [
            "Yes",
            "No"
        ]
    )

    internship = st.selectbox(
        "Internship",
        [
            "Yes",
            "No"
        ]
    )

    training = st.selectbox(
        "Training",
        [
            "Yes",
            "No"
        ]
    )


with col2:

    language_count = st.number_input(
        "Programming Languages",
        min_value=0,
        max_value=15,
        value=2
    )

    certification_count = st.number_input(
        "Certifications",
        min_value=0,
        max_value=15,
        value=1
    )


st.divider()


# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader(
    "Model Comparison"
)

st.write(
    "Performance of all classification models "
    "on the same held-out test set."
)


# Make a copy only for displaying percentages
display_metrics = metrics_df.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]:

    display_metrics[column] = (
        display_metrics[column]
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )


# THIS TABLE IS GENERATED FROM THE ACTUAL
# MODEL EVALUATION RESULTS.
st.dataframe(
    display_metrics,
    hide_index=True,
    use_container_width=True
)


st.caption(
    "Metrics are calculated automatically from "
    "the 20% held-out test set. No metric values "
    "are hardcoded."
)


st.divider()


# ==========================================
# PREDICTION
# ==========================================

st.subheader(
    "Generate Prediction"
)


if st.button(
    "Predict Placement",
    type="primary",
    use_container_width=True
):

    student = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "course": course,
                "branch": branch,
                "year": year,
                "cgpa": cgpa,
                "backlogs": backlogs,
                "projects": projects,
                "tech_skill": tech_skill,
                "comm_skill": comm_skill,
                "apt_skill": apt_skill,
                "team_skill": team_skill,
                "participation": participation,
                "internship": internship,
                "training": training,
                "language_count": language_count,
                "certification_count": certification_count
            }
        ]
    )


    # Generate prediction
    prediction = int(
        prediction_model.predict(
            student
        )[0]
    )


    # Generate probability
    probabilities = (
        prediction_model.predict_proba(
            student
        )[0]
    )


    probability = float(
        probabilities[prediction]
    )


    st.divider()


    # ======================================
    # PREDICTION RESULT
    # ======================================

    st.subheader(
        "Prediction Result"
    )


    if prediction == 1:

        st.success(
            "🟢 PLACED"
        )

    else:

        st.error(
            "🔴 NOT PLACED"
        )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Probability",
            f"{probability:.1%}"
        )


    with result_col2:

        st.metric(
            "Model",
            "Random Forest"
        )


    st.warning(
        "Model reliability is limited. "
        "Validation testing showed weak predictive "
        "signal in the supplied synthetic dataset."
    )


# ==========================================
# INTERPRETATION
# ==========================================

st.divider()

st.subheader(
    "Interpretation"
)

st.write(
    "The prediction should be treated as an experimental "
    "ML output rather than a reliable placement forecast. "
    "The supplied synthetic dataset contains weak "
    "relationships between the available student features "
    "and placement outcome."
)