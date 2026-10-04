import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# 1. LOAD DATA


df = pd.read_csv("placement_readiness_synthetic_10000.csv")

print("Original shape:", df.shape)


# 2. REMOVE DUPLICATES

df = df.drop_duplicates().reset_index(drop=True)

print("After removing duplicates:", df.shape)


# 3. CLEAN BACKLOGS

df["backlogs"] = df["backlogs"].fillna("0")

df["backlogs"] = df["backlogs"].replace({
    "3+": 3
})

df["backlogs"] = df["backlogs"].astype(int)


# 4. CREATE LANGUAGE COUNT

def count_languages(value):
    if pd.isna(value):
        return 0

    return len(value.split(";"))


df["language_count"] = df["languages"].apply(count_languages)


# 5. CREATE CERTIFICATION COUNT

def count_certifications(value):
    if pd.isna(value):
        return 0

    return len(value.split(";"))


df["certification_count"] = df["certifications"].apply(
    count_certifications
)


# 6. DROP IRRELEVANT COLUMNS

df = df.drop(columns=[
    "name",
    "email",
    "other_course",
    "opinion",
    "languages",
    "certifications"
])


# 7. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["placement"])

y = df["placement"].map({
    "Not Placed": 0,
    "Placed": 1
})


# 8. DEFINE COLUMN TYPES

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


# 9. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 10. PREPROCESSING

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# 11. LOGISTIC REGRESSION

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)


# 12. DECISION TREE

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)


print("\nModels trained successfully!")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 13. EVALUATE LOGISTIC REGRESSION
# ==========================================

print("\n===== LOGISTIC REGRESSION =====")

print("Accuracy:",
      accuracy_score(y_test, logistic_predictions))

print("Precision:",
      precision_score(y_test, logistic_predictions))

print("Recall:",
      recall_score(y_test, logistic_predictions))

print("F1 Score:",
      f1_score(y_test, logistic_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))


# ==========================================
# 14. EVALUATE DECISION TREE
# ==========================================

print("\n===== DECISION TREE =====")

print("Accuracy:",
      accuracy_score(y_test, tree_predictions))

print("Precision:",
      precision_score(y_test, tree_predictions))

print("Recall:",
      recall_score(y_test, tree_predictions))

print("F1 Score:",
      f1_score(y_test, tree_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_predictions))

print("\n===== DATASET SANITY CHECK =====")

print("\nAverage CGPA by placement:")
print(df.groupby("placement")["cgpa"].mean())

print("\nAverage projects by placement:")
print(df.groupby("placement")["projects"].mean())

print("\nAverage skill scores by placement:")
print(
    df.groupby("placement")[
        ["tech_skill", "comm_skill", "apt_skill", "team_skill"]
    ].mean()
)

print("\nInternship vs placement:")
print(pd.crosstab(df["internship"], df["placement"], normalize="index"))

print("\nTraining vs placement:")
print(pd.crosstab(df["training"], df["placement"], normalize="index"))

print("\nParticipation vs placement:")
print(pd.crosstab(df["participation"], df["placement"], normalize="index"))

print("\nBranch vs placement:")
print(pd.crosstab(df["branch"], df["placement"], normalize="index"))

print("\nYear vs placement:")
print(pd.crosstab(df["year"], df["placement"], normalize="index"))

print("\n===== NUMERIC CORRELATIONS =====")

numeric_check = df[
    [
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
].copy()

numeric_check["placement"] = y

print(
    numeric_check.corr()["placement"]
    .sort_values(ascending=False)
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ==========================================
# 15. RANDOM FOREST
# ==========================================

random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    ))
])

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)


print("\n===== RANDOM FOREST =====")

print("Accuracy:",
      accuracy_score(y_test, rf_predictions))

print("Precision:",
      precision_score(y_test, rf_predictions))

print("Recall:",
      recall_score(y_test, rf_predictions))

print("F1 Score:",
      f1_score(y_test, rf_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


# ==========================================
# 16. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("\n===== CROSS VALIDATION =====")
print("F1 scores:", cv_scores)
print("Mean F1:", cv_scores.mean())


from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

print("\n===== FEATURE SIGNAL CHECK =====")

# Transform the data using our existing preprocessor
X_transformed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Mutual information
mi_scores = mutual_info_classif(
    X_transformed,
    y,
    random_state=42
)

mi_results = pd.DataFrame({
    "feature": feature_names,
    "mutual_information": mi_scores
}).sort_values(
    "mutual_information",
    ascending=False
)

print("\nTop 20 features by mutual information:")
print(mi_results.head(20).to_string(index=False))

print("\n===== SINGLE FEATURE TEST =====")

for column in X.columns:

    if X[column].dtype == "object":

        temp = pd.get_dummies(
            X[[column]],
            drop_first=False
        )

    else:

        temp = X[[column]].copy()

        temp = temp.fillna(temp.median())

    model = LogisticRegression(max_iter=1000)

    model.fit(temp, y)

    score = model.score(temp, y)

    print(f"{column:25} {score:.3f}")

    # ==========================================
# 17. EXAMPLE PREDICTION
# ==========================================

def predict_student(student_data):

    student_df = pd.DataFrame([student_data])

    prediction = random_forest.predict(student_df)[0]

    probability = random_forest.predict_proba(student_df)[0]

    if prediction == 1:
        result = "Placed"
        confidence = probability[1]
    else:
        result = "Not Placed"
        confidence = probability[0]

    return result, confidence


# Example student
example_student = {
    "gender": "Male",
    "age": 21,
    "course": "B.Tech",
    "branch": "CSE",
    "year": "4th Year",
    "cgpa": 8.5,
    "backlogs": 0,
    "projects": 5,
    "tech_skill": 4,
    "comm_skill": 4,
    "apt_skill": 5,
    "team_skill": 4,
    "participation": "Yes",
    "internship": "Yes",
    "training": "Yes",
    "language_count": 3,
    "certification_count": 2
}

prediction, confidence = predict_student(example_student)

print("\n===== EXAMPLE PREDICTION =====")
print("Prediction:", prediction)
print("Confidence:", f"{confidence:.2%}")