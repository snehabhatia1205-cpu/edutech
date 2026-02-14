import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EdTech Engagement Analyzer", layout="wide")

st.title("📊 EdTech Student Engagement & Dropout Predictor")

# ----------------------------
# Generate Synthetic Dataset
# ----------------------------

def generate_dataset(n=1000):
    np.random.seed(42)

    subjects = ["Data Science", "Mathematics", "Programming", "Business", "AI", "Cyber Security"]
    
    data = {
        "student_id": np.arange(1, n+1),
        "course_id": np.random.randint(100, 200, n),
        "subject": np.random.choice(subjects, n),
        "engagement_hours": np.random.normal(30, 10, n).clip(1),
        "weekly_logins": np.random.randint(1, 15, n),
        "assignment_submission_rate": np.random.uniform(0.3, 1.0, n),
        "assessment_score": np.random.normal(70, 15, n).clip(30, 100),
    }

    df = pd.DataFrame(data)

    # Create text column for regex extraction
    df["engagement_text"] = df["engagement_hours"].round(1).astype(str) + " hours"

    # Completion logic
    df["completion_status"] = np.where(
        (df["engagement_hours"] > 25) &
        (df["assignment_submission_rate"] > 0.6) &
        (df["assessment_score"] > 60),
        1, 0
    )

    # Introduce missing values
    df.loc[df.sample(frac=0.05).index, "engagement_hours"] = np.nan
    df.loc[df.sample(frac=0.05).index, "assignment_submission_rate"] = np.nan

    return df


# Sidebar options
st.sidebar.header("Options")
generate_data = st.sidebar.button("Generate Synthetic Dataset (1000 rows)")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset")

# Load dataset
if generate_data:
    df = generate_dataset()
    st.success("Synthetic dataset generated with 1000 rows.")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")
else:
    st.info("Please generate or upload a dataset.")
    st.stop()

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Data Cleaning
# ----------------------------

st.subheader("Data Cleaning")

# Fill missing values
df["engagement_hours"].fillna(df["engagement_hours"].mean(), inplace=True)
df["assignment_submission_rate"].fillna(df["assignment_submission_rate"].mean(), inplace=True)

st.write("Missing values handled using mean imputation.")

# ----------------------------
# Regular Expression Feature Extraction
# ----------------------------

st.subheader("Regex Transformation")

def extract_hours(text):
    match = re.search(r"(\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else 0

df["extracted_engagement_hours"] = df["engagement_text"].apply(extract_hours)

st.write("Extracted numeric engagement hours from text column using regex.")

# ----------------------------
# Feature Engineering
# ----------------------------

st.subheader("Feature Engineering")

df["engagement_per_login"] = df["engagement_hours"] / df["weekly_logins"]
df["performance_index"] = (
    df["assignment_submission_rate"] * 0.4 +
    df["assessment_score"] * 0.6
)

# Encode subject
df = pd.get_dummies(df, columns=["subject"], drop_first=True)

st.write("Created engagement_per_login and performance_index features.")

# ----------------------------
# Normalization
# ----------------------------

st.subheader("Normalization")

numerical_cols = [
    "engagement_hours",
    "weekly_logins",
    "assignment_submission_rate",
    "assessment_score",
    "engagement_per_login",
    "performance_index"
]

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

st.write("Numerical features normalized using MinMaxScaler.")

# ----------------------------
# Modeling
# ----------------------------

st.subheader("Dropout Prediction Model")

X = df.drop(columns=["student_id", "course_id", "completion_status", "engagement_text"])
y = df["completion_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ----------------------------
# Feature Importance
# ----------------------------

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(data=importance.head(10), x="Importance", y="Feature", ax=ax)
st.pyplot(fig)

st.success("Analysis Complete ✅")
