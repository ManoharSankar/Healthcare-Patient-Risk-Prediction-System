import streamlit as st
import pandas as pd
import joblib
import boto3
import io
import json
import logging
import watchtower
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
import plotly.express as px

# --------------------------
# 🎯 Streamlit Config
# --------------------------
st.set_page_config(page_title="🏥 Patient Risk Predictor", page_icon="💉", layout="wide")

st.markdown("""
    <style>
        body { background-color: #f8fafc; }
        .main { padding: 1rem 2rem; }
        .title { font-size: 28px; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; }
        .subheader { color: #0072B1; font-weight: 600; font-size: 20px; }
        .risk-box {
            border-radius: 10px; padding: 1.5rem; margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 📡 CloudWatch Logging Setup
# --------------------------
@st.cache_resource
def get_logger():
    """Initialize CloudWatch logger"""
    logger = logging.getLogger("patient-risk-app")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        session = boto3.Session(region_name="ap-south-1")
        cw_handler = watchtower.CloudWatchLogHandler(
            boto3_session=session,
            log_group="/aws/streamlit/patient-risk",
            stream_name="app-logs"
        )
        formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        cw_handler.setFormatter(formatter)
        logger.addHandler(cw_handler)

    return logger

# ✅ Initialize logger immediately so it's available globally
logger = get_logger()
logger.info("✅ Streamlit app started successfully.")
# --------------------------
# 🧠 Load Model from S3
# --------------------------
@st.cache_resource
def load_model_from_s3(bucket="healthcarepatientrecords25"):
    s3 = boto3.client("s3")
    try:
        model_obj = s3.get_object(Bucket=bucket, Key="models/patient_risk_model.pkl")
        cols_obj = s3.get_object(Bucket=bucket, Key="models/model_columns.pkl")
        model = joblib.load(io.BytesIO(model_obj["Body"].read()))
        model_columns = joblib.load(io.BytesIO(cols_obj["Body"].read()))
        logger.info("✅ Model loaded successfully from S3.")
        return model, model_columns
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        st.error(f"Model could not be loaded from S3: {e}")
        st.stop()

model, model_columns = load_model_from_s3()

# --------------------------
# 🔐 RDS Connection
# --------------------------
@st.cache_resource
def get_engine():
    secret_name = "patientriskdb/credentials"
    region_name = "ap-south-1"
    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        secret = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
        db_name = secret.get("dbname", secret.get("dbClusterIdentifier"))
        engine_type = secret.get("engine", "postgresql").replace("postgres", "postgresql")
        engine = create_engine(f"{engine_type}://{secret['username']}:{secret['password']}@{secret['host']}/{db_name}")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful.")
        return engine
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        st.error(f"Database connection failed: {e}")
        st.stop()

engine = get_engine()

# --------------------------
# 🧩 Ensure Table Schema Exists
# --------------------------
def ensure_table_schema(engine):
    """Ensure the patient_predictions table has all required columns"""
    required_columns = {
        "patient_id": "VARCHAR(50)",
        "age": "INT",
        "heart_rate": "INT",
        "bp_systolic": "INT",
        "bp_diastolic": "INT",
        "hemoglobin": "FLOAT",
        "length_of_stay": "INT",
        "gender": "VARCHAR(50)",
        "race": "VARCHAR(50)",
        "diagnosis": "VARCHAR(100)",
        "risk_score": "FLOAT",
        "prediction_label": "VARCHAR(50)",
        "timestamp": "TIMESTAMP"
    }

    inspector = inspect(engine)
    with engine.connect() as conn:
        tables = inspector.get_table_names()
        if "patient_predictions" not in tables:
            # Create table if missing
            cols_sql = ", ".join([f"{col} {dtype}" for col, dtype in required_columns.items()])
            conn.execute(text(f"CREATE TABLE patient_predictions ({cols_sql});"))
            logger.info("🆕 Created patient_predictions table in RDS.")
        else:
            # Add any missing columns
            existing_cols = [col["name"] for col in inspector.get_columns("patient_predictions")]
            for col, dtype in required_columns.items():
                if col not in existing_cols:
                    conn.execute(text(f"ALTER TABLE patient_predictions ADD COLUMN {col} {dtype};"))
                    logger.info(f"➕ Added missing column '{col}' to patient_predictions table.")

ensure_table_schema(engine)

# --------------------------
# 🧮 Sidebar
# --------------------------
menu = st.sidebar.radio("Navigation", ["Predict Risk", "Analytics Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("AI model powered by RandomForest & AWS")

# --------------------------
# 💉 PREDICT RISK
# --------------------------
if menu == "Predict Risk":
    st.markdown('<p class="title">💉 Patient Readmission Risk Prediction</p>', unsafe_allow_html=True)
    st.caption("Enter patient data below to estimate readmission risk.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 120, 45)
            heart_rate = st.number_input("Heart Rate", 30, 200, 78)
        with c2:
            bp_systolic = st.number_input("Systolic BP", 80, 200, 125)
            bp_diastolic = st.number_input("Diastolic BP", 40, 120, 82)
        with c3:
            hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
            length_of_stay = st.number_input("Hospital Stay (days)", 0, 60, 4)

        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
        diagnosis = st.selectbox("Diagnosis", ["Diabetes", "Heart Disease", "Kidney Disease", "Other"])
        patient_id = st.text_input("Patient ID", value="P001")

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        try:
            input_data = pd.DataFrame({
                "age": [age],
                "heart_rate": [heart_rate],
                "bp_systolic": [bp_systolic],
                "bp_diastolic": [bp_diastolic],
                "hemoglobin": [hemoglobin],
                "length_of_stay": [length_of_stay],
                "gender": [gender],
                "race": [race],
                "diagnosis": [diagnosis],
            })
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[:, 1][0]
            label = "High Risk" if pred == 1 else "Low Risk"

            color = "#ffe5e5" if pred == 1 else "#e8f8f5"
            icon = "⚠️" if pred == 1 else "✅"
            st.markdown(f"""
                <div class="risk-box" style="background-color:{color}">
                    <h3>{icon} {label}</h3>
                    <p>Probability: {prob*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(prob)
            logger.info(f"Prediction made for {patient_id}: {label} ({prob:.3f})")

            record = pd.DataFrame([{
                "patient_id": patient_id,
                "age": age,
                "heart_rate": heart_rate,
                "bp_systolic": bp_systolic,
                "bp_diastolic": bp_diastolic,
                "hemoglobin": hemoglobin,
                "length_of_stay": length_of_stay,
                "gender": gender,
                "race": race,
                "diagnosis": diagnosis,
                "risk_score": float(prob),
                "prediction_label": label,
                "timestamp": pd.Timestamp.now(tz="UTC")
            }])

            record.to_sql("patient_predictions", engine, if_exists="append", index=False)
            st.success("💾 Prediction saved to database.")
            logger.info(f"✅ Saved prediction for {patient_id} to RDS.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction failed for {patient_id}: {e}")

# --------------------------
# 📊 ANALYTICS DASHBOARD
# --------------------------
elif menu == "Analytics Dashboard":
    st.markdown('<p class="title">📈 Patient Risk Analytics</p>', unsafe_allow_html=True)

    @st.cache_data(ttl=180)
    def fetch_data():
        query = """
        SELECT patient_id, risk_score, prediction_label, timestamp
        FROM patient_predictions
        ORDER BY timestamp DESC
        LIMIT 500
        """
        return pd.read_sql(query, engine)

    with st.spinner("Fetching analytics data..."):
        try:
            df = fetch_data()
            logger.info(f"Fetched {len(df)} records for analytics.")
        except Exception as e:
            st.error(f"❌ Database error: {e}")
            logger.error(f"Analytics fetch error: {e}")
            st.stop()

    if df.empty:
        st.info("No prediction records found.")
    else:
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        col1.metric("📊 Total Predictions", len(df))
        col2.metric("⚠️ High Risk", len(df[df["prediction_label"] == "High Risk"]))
        col3.metric("✅ Low Risk", len(df[df["prediction_label"] == "Low Risk"]))
        col4.metric("📉 Avg Risk Score", f"{df['risk_score'].mean():.2f}")

        fig = px.line(
            df, x="timestamp", y="risk_score", color="prediction_label",
            title="Risk Trends Over Time", markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Recent Predictions"):
            st.dataframe(df, use_container_width=True)

        if st.button("🔄 Refresh Data"):
            fetch_data.clear()
            st.experimental_rerun()
