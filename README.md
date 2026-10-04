# gdg-projects-kkc
THIS IS AI GENERATED I AM SORRY.


# Placement Readiness Predictor

A machine learning project that predicts whether a student is likely to be placed based on academic performance, skills, projects, and experience-related features.

## Objective

The objective of this project is to build a binary classification model using the placement-readiness dataset provided by GDG-USAR.

The project includes data cleaning, feature preparation, categorical encoding, model comparison, evaluation, and a simple Streamlit interface for making predictions.

## Dataset

The project uses the provided:

`placement_readiness_synthetic_10000.csv`

The dataset contains student-related academic, skill, and experience information along with a binary placement outcome.

The original dataset contained 10,000 rows. After removing duplicate records, 1,000 unique records remained.

The target variable is:

- `Placed`
- `Not Placed`

## Data Preprocessing

The following preprocessing steps were applied:

### Duplicate removal

Duplicate records were removed before model training.

### Backlogs

Missing backlog values were treated as zero.

The value `3+` was converted to `3` so that the feature could be treated numerically.

### Languages

The original `languages` field contained multiple programming languages separated by semicolons.

Instead of using the raw text directly, a new feature called `language_count` was created.

### Certifications

The original `certifications` field was similarly converted into a numeric `certification_count` feature.

### Categorical features

Categorical features were handled using:

- Most-frequent imputation for missing categorical values
- One-Hot Encoding

### Numerical features

Numerical features were handled using:

- Median imputation for missing values
- Standard scaling

### Removed columns

The following columns were not used as predictive features:

- `name`
- `email`
- `other_course`
- `opinion`
- `languages`
- `certifications`

These columns were either identifiers, largely missing, free-text information, or were replaced by more useful derived features.

## Features Used

### Numerical features

- Age
- CGPA
- Backlogs
- Projects
- Technical Skill
- Communication Skill
- Aptitude Skill
- Team Skill
- Language Count
- Certification Count

### Categorical features

- Gender
- Course
- Branch
- Year
- Participation
- Internship
- Training

## Models

Three classification models were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

All models use the same preprocessing pipeline so that their results can be compared fairly.

## Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

A fixed random state of 42 and stratified splitting were used to make the evaluation reproducible while maintaining the class distribution.

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The Streamlit interface calculates these metrics automatically from predictions on the held-out test set.

## Results

The model comparison is displayed directly in the Streamlit application.

The supplied synthetic dataset produced relatively weak predictive performance. This suggests that the available features have limited predictive relationship with the placement target.

Rather than artificially modifying the data or target labels to obtain higher scores, the project reports the observed results honestly.

## Prediction Interface

A Streamlit interface was created where a user can enter:

- Academic information
- Skills
- Number of projects
- Internship information
- Training
- Participation
- Programming language count
- Certification count

The application then generates a placement prediction using the Random Forest model.

The interface also displays the predicted probability.

## How to Run

Install the required Python packages:


pip install pandas numpy scikit-learn streamlit

streamlit run "predictor ui.py"

#Limitations

The dataset is synthetic and contains weak relationships between the available features and the placement outcome.

Therefore, the model should be treated as a demonstration of a complete machine learning pipeline rather than as a reliable real-world placement prediction system.

The predicted probability should also not be interpreted as a guaranteed probability of actual placement.

Conclusion

This project demonstrates the complete workflow of a binary classification problem:


Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Evaluation
   ↓
Prediction Interface
