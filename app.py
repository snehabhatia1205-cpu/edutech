import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EdTech Student Engagement Dashboard", layout="wide")
st.title("📊 EdTech Student Engagement & Dropout Predictor")

# -----------------------------
# Upload or Generate Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")
else:
    st.info("No dataset uploaded. Generating synthetic dataset with 1000 rows.")

    n = 1000
    data = []
    first_names = ["John", "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia"]
    last_names = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas"]

    np.random.seed(42)
    for i in range(1, n+1):
        engagement_hours = round(np.random.normal(30, 10), 2)
        assessment_score = round(np.random.normal(70, 15), 2)
        videos_pct = round(np.random.uniform(40, 100), 2)
        assignments = np.random.randint(1, 12)
        quizzes = np.random.randint(1, 10)
        login_frequency = np.random.randint(5, 50)
        session_duration = round(np.random.uniform(10, 120), 2)
        inactivity_days = np.random.randint(0, 30)
        completion_status = 1 if (engagement_hours + assessment_score/2) > 60 else 0
        name = random.choice(first_names) + " " + random.choice(last_names)

        row = {
            "student_id": f"STU_{i:04d}",
            "course_id": random.choice(["C_DS_01", "C_AI_02", "C_ML_03", "C_WEB_04"]),
            "student_name": name,
            "email": f"student{i}@edtech.com",
            "device_type": random.choice(["Mobile", "Desktop", "Tablet"]),
            "learning_path_type": random.choice(["Self-Paced", "Instructor-Led"]),
            "engagement_hours": engagement_hours,
            "assessment_score_avg": assessment_score,
            "videos_watched_pct": videos_pct,
            "assignments_submitted": assignments,
            "quizzes_attempted": quizzes,
            "login_frequency": login_frequency,
            "session_duration_avg": session_duration,
            "inactivity_gap_days": inactivity_days,
            "completion_status": completion_status
        }

        data.append(row)

    df = pd.DataFrame(data)

    # Introduce some missing values
    for col in ["engagement_hours", "assessment_score_avg"]:
        df.loc[df.sample(frac=0.02).index, col] = np.nan

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Ensure Required Columns Exist
# -----------------------------
if "assignment_submission_rate" not in df.columns:
    df["assignment_submission_rate"] = df["assignments_submitted"] / df["assignments_submitted"].max()

if "performance_index" not in df.columns:
    df["performance_index"] = 0.4*df["assessment_score_avg"] + \
                              0.3*df["videos_watched_pct"] + \
                              0.3*df["engagement_hours"]

if "at_risk" not in df.columns:
    df["at_risk"] = np.where((df["engagement_hours"] < 20) & (df["assessment_score_avg"] < 60), 1, 0)

# -----------------------------
# Data Cleaning
# -----------------------------
st.subheader("Data Cleaning")
df = df.drop_duplicates()
for col in ["engagement_hours", "assessment_score_avg", "assignment_submission_rate", "performance_index"]:
    df[col].fillna(df[col].mean(), inplace=True)

st.write("✅ Duplicates removed and missing values handled.")

# -----------------------------
# Regex Processing
# -----------------------------
st.subheader("Regex Processing")
if "email" in df.columns:
    df["email_domain"] = df["email"].apply(
        lambda x: re.search(r'@([\w\.]+)', str(x)).group(1) if re.search(r'@([\w\.]+)', str(x)) else None
    )
    st.write("✅ Email domain extracted using regex.")

# -----------------------------
# Normalization
# -----------------------------
st.subheader("Normalization")
numeric_cols = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
st.write("✅ Numeric features normalized.")

# -----------------------------
# Insights
# -----------------------------
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric("Completion Rate (%)", round(df["completion_status"].mean()*100, 2))
col3.metric("At Risk Students", df["at_risk"].sum())

# -----------------------------
# Show Processed Dataset
# -----------------------------
st.subheader("Processed Dataset (Normalized)")
st.dataframe(df_scaled.head())

# -----------------------------
# Download Processed CSV
# -----------------------------
csv = df_scaled.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Processed Dataset",
    data=csv,
    file_name="processed_student_data.csv",
    mime="text/csv"
)

st.success("✅ App is ready!")
