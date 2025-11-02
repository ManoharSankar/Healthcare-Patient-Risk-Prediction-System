# ==========================================================
# 🏥 Patient Readmission Predictor (SageMaker + RDS + CloudWatch)
# ==========================================================
import streamlit as st
import pandas as pd
import boto3
import json
from datetime import datetime
from sqlalchemy import create_engine, text
import plotly.express as px

# --------------------------------------------------
# ⚙️ CONFIGURATION
# --------------------------------------------------
REGION = "ap-south-1"
ENDPOINT_NAME = "healthcarepredict"
SECRET_NAME = "patientriskdb/credentials"
LOG_GROUP = "/sagemaker/healthcarepredict/app-logs"
SNS_TOPIC_ARN = "arn:aws:sns:ap-south-1:051826708114:patientriskalert"

# --------------------------------------------------
# ☁️ AWS CLIENTS
# --------------------------------------------------
logs_client = boto3.client("logs", region_name=REGION)
runtime_client = boto3.client("sagemaker-runtime", region_name=REGION)
secrets_client = boto3.client("secretsmanager", region_name=REGION)
sns_client = boto3.client("sns", region_name=REGION)

# --------------------------------------------------
# 🪵 CLOUDWATCH LOGGING
# --------------------------------------------------
def log_to_cloudwatch(message, level="INFO"):
    """Send a log message to CloudWatch."""
    try:
        logs_client.create_log_group(logGroupName=LOG_GROUP)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass

    stream_name = "streamlit-app"
    try:
        logs_client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=stream_name)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass

    log_event = {
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
        "message": f"[{level}] {message}"
    }

    try:
        logs_client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=stream_name,
            logEvents=[log_event]
        )
    except Exception as e:
        print("Log failed:", e)

# --------------------------------------------------
# 🔮 SAGEMAKER PREDICTION FUNCTION
# --------------------------------------------------
def predict_with_sagemaker(payload):
    """Send CSV payload to SageMaker endpoint and return prediction."""
    try:
        csv_data = ",".join(str(x) for x in payload.values())
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_data
        )
        result = response["Body"].read().decode("utf-8").strip()
        log_to_cloudwatch(f"Prediction success: {result}")
        return [float(result)]
    except Exception as e:
        log_to_cloudwatch(f"Prediction failed: {e}", "ERROR")
        st.error(f"❌ Prediction failed: {e}")
        return None

# --------------------------------------------------
# 🗄️ DATABASE CONNECTION
# --------------------------------------------------
@st.cache_resource
def get_engine():
    try:
        secret = json.loads(
            secrets_client.get_secret_value(SecretId=SECRET_NAME)["SecretString"]
        )
        db_name = secret.get("dbname", secret.get("dbClusterIdentifier"))
        engine_type = secret.get("engine", "postgresql").replace("postgres", "postgresql")

        engine = create_engine(
            f"{engine_type}://{secret['username']}:{secret['password']}@{secret['host']}/{db_name}"
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

engine = get_engine()

# --------------------------------------------------
# 🧱 ENSURE TABLE EXISTS
# --------------------------------------------------
def ensure_table(engine):
    create_sql = """
    CREATE TABLE IF NOT EXISTS patient_predictions (
        patient_id VARCHAR(50),
        time_in_hospital INT,
        num_medications INT,
        num_lab_procedures INT,
        num_procedures INT,
        age INT,
        number_diagnoses INT,
        heart_rate INT,
        hemoglobin FLOAT,
        length_of_stay INT,
        risk_score FLOAT,
        prediction_label VARCHAR(50),
        timestamp TIMESTAMP
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))
    log_to_cloudwatch("✅ Verified patient_predictions table")

ensure_table(engine)

# --------------------------------------------------
# 🎨 STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="🏥 Patient Readmission Risk", page_icon="💉", layout="wide")
st.title("🏥 Patient Readmission Risk Predictor")

menu = st.sidebar.radio("Navigation", ["🔮 Predict Risk", "📊 Dashboard", "📈 Risk Analysis"])
st.sidebar.info("Powered by AWS SageMaker + CloudWatch + RDS")

# --------------------------------------------------
# 💉 PREDICT PAGE
# --------------------------------------------------
if menu == "🔮 Predict Risk":
    st.subheader("🧍 Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "P001")
        time_in_hospital = st.number_input("Time in Hospital", 0, 30, 5)
        num_medications = st.number_input("Number of Medications", 0, 100, 10)
        num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 200, 40)
        num_procedures = st.number_input("Number of Procedures", 0, 20, 2)
    with col2:
        age = st.number_input("Age", 0, 120, 45)
        number_diagnoses = st.number_input("Number of Diagnoses", 0, 20, 5)
        heart_rate = st.number_input("Heart Rate", 40, 200, 80)
        hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
        length_of_stay = st.number_input("Length of Stay", 0, 60, 3)

    if st.button("🚀 Predict Risk"):
        model_input = {
            "time_in_hospital": time_in_hospital,
            "num_medications": num_medications,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "age": age,
            "number_diagnoses": number_diagnoses
        }

        result = predict_with_sagemaker(model_input)
        if result:
            prob = float(result[0])
            label = "High Risk" if prob >= 0.5 else "Low Risk"

            st.success(f"Prediction: **{label}** (Risk Score: {prob:.3f})")
            st.progress(prob)

            record = pd.DataFrame([{
                "patient_id": patient_id,
                "time_in_hospital": time_in_hospital,
                "num_medications": num_medications,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "age": age,
                "number_diagnoses": number_diagnoses,
                "heart_rate": heart_rate,
                "hemoglobin": hemoglobin,
                "length_of_stay": length_of_stay,
                "risk_score": prob,
                "prediction_label": label,
                "timestamp": datetime.utcnow()
            }])

            record.to_sql("patient_predictions", engine, if_exists="append", index=False)
            log_to_cloudwatch(f"Prediction saved for {patient_id}")

            if label == "High Risk":
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject="🚨 High Risk Patient Detected",
                    Message=(
                        f"Patient ID: {patient_id}\n"
                        f"Risk Score: {prob:.3f}\n"
                        f"Prediction: {label}\n"
                        f"Check dashboard for details."
                    )
                )
                st.warning("🚨 Alert sent for high-risk patient!")

# --------------------------------------------------
# 📊 DASHBOARD PAGE
# --------------------------------------------------
elif menu == "📊 Dashboard":
    st.subheader("📊 Prediction Dashboard")

    try:
        df = pd.read_sql("SELECT * FROM patient_predictions ORDER BY timestamp DESC LIMIT 200;", engine)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        log_to_cloudwatch(f"DB fetch failed: {e}", "ERROR")
        st.stop()

    if df.empty:
        st.info("No predictions yet.")
    else:
        st.metric("Total Predictions", len(df))
        st.metric("High Risk Patients", len(df[df["prediction_label"] == "High Risk"]))
        st.metric("Average Risk Score", round(df["risk_score"].mean(), 3))
        st.dataframe(df)

# --------------------------------------------------
# 📈 RISK ANALYSIS PAGE
# --------------------------------------------------
elif menu == "📈 Risk Analysis":
    st.subheader("📈 Patient Risk Analysis")

    try:
        df = pd.read_sql("SELECT * FROM patient_predictions;", engine)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        log_to_cloudwatch(f"DB fetch failed: {e}", "ERROR")
        st.stop()

    if df.empty:
        st.info("No predictions yet.")
    else:
        # 🔹 Risk Distribution
        st.markdown("### 🔹 Risk Distribution")
        fig1 = px.histogram(df, x="risk_score", color="prediction_label", nbins=20,
                            title="Distribution of Risk Scores")
        st.plotly_chart(fig1, use_container_width=True)

        # 🔹 Risk Over Time
        st.markdown("### 🔹 Risk Over Time")
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        trend = df.groupby("date")["risk_score"].mean().reset_index()
        fig2 = px.line(trend, x="date", y="risk_score", title="Average Risk Score Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        # 🔹 Feature Insights
        st.markdown("### 🔹 Feature Insights")
        colA, colB = st.columns(2)
        with colA:
            fig3 = px.box(df, x="prediction_label", y="age", title="Age vs Risk Level")
            st.plotly_chart(fig3, use_container_width=True)
        with colB:
            fig4 = px.box(df, x="prediction_label", y="hemoglobin", title="Hemoglobin vs Risk Level")
            st.plotly_chart(fig4, use_container_width=True)

        # 🌟 Feature Importance
        st.markdown("### 🌟 Feature Importance Analysis")
        try:
            features = ["time_in_hospital", "num_medications", "num_lab_procedures",
                        "num_procedures", "age", "number_diagnoses",
                        "heart_rate", "hemoglobin", "length_of_stay"]
            corr = df[features + ["risk_score"]].corr()["risk_score"].drop("risk_score").abs()
            importance_df = pd.DataFrame({"Feature": corr.index, "Importance": corr.values})
            importance_df.sort_values("Importance", ascending=False, inplace=True)

            fig5 = px.bar(importance_df, x="Importance", y="Feature",
                          orientation="h", title="Approximate Feature Importance",
                          color="Importance", color_continuous_scale="Bluered")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not compute feature importance: {e}")

        st.caption("📘 Note: Feature importance is approximated using correlation with model predictions.")