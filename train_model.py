import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import io

# --- AWS S3 CONFIG ---
S3_BUCKET = "healthcarepatientrecords25"
S3_INPUT_KEY = "raw_dataset/patient_data.csv"
S3_MODEL_KEY = "models/patient_risk_model.pkl"
S3_SCALER_KEY = "models/scaler.pkl"
S3_COLUMNS_KEY = "models/model_columns.pkl"

# ---------------------------
# 1. LOAD DATA FROM S3
# ---------------------------
def load_data_from_s3():
    """Load dataset from AWS S3 bucket"""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_INPUT_KEY)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print("✅ Data loaded from S3 successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading data from S3: {e}")
        raise


# ---------------------------
# 2. PREPROCESS DATA
# ---------------------------
def preprocess_data(df):
    """Handle missing values, encode categorical columns, and scale numeric features"""
    # Fill missing numeric and categorical values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=False)

    # 🎯 Handle target column dynamically
    if "readmitted" in df.columns:
    # 1 = Readmitted (High Risk), 0 = Not Readmitted (Low Risk)
        y = (df["readmitted"] != "NO").astype(int)
        X = df.drop("readmitted", axis=1)
        X = pd.get_dummies(X, drop_first=False)

    elif "readmitted_NO" in df_encoded.columns:
    # Use existing encoded column
        y = 1 - df_encoded["readmitted_NO"]  # Flip so that 1 = Readmitted
        X = df_encoded.drop("readmitted_NO", axis=1)

    else:
        raise ValueError("Target column 'readmitted' or 'readmitted_NO' not found.")


    # Scale only numeric columns
    num_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    print("✅ Data preprocessed successfully!")
    return X, y, scaler


# ---------------------------
# 3. UPLOAD FILE TO S3
# ---------------------------
def upload_to_s3(file_obj, key):
    """Upload file object to S3"""
    try:
        s3 = boto3.client("s3")
        s3.upload_fileobj(file_obj, S3_BUCKET, key)
        print(f"📦 Uploaded to S3: s3://{S3_BUCKET}/{key}")
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        raise


# ---------------------------
# 4. TRAIN AND SAVE MODEL
# ---------------------------
def train_and_save_model(X, y, scaler):
    """Train model and save to S3"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("✅ Model trained successfully!")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 🔹 Save model, scaler, and columns
    model_columns = list(X.columns)

    # Save model to bytes
    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes, compress=3)
    model_bytes.seek(0)

    # Save scaler to bytes
    scaler_bytes = io.BytesIO()
    joblib.dump(scaler, scaler_bytes, compress=3)
    scaler_bytes.seek(0)

    # Save model columns to bytes
    columns_bytes = io.BytesIO()
    joblib.dump(model_columns, columns_bytes, compress=3)
    columns_bytes.seek(0)

    # ✅ Upload to S3 (correctly mapped)
    upload_to_s3(model_bytes, S3_MODEL_KEY)
    upload_to_s3(scaler_bytes, S3_SCALER_KEY)
    upload_to_s3(columns_bytes, S3_COLUMNS_KEY)


# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    try:
        df = load_data_from_s3()
        X, y, scaler = preprocess_data(df)
        train_and_save_model(X, y, scaler)
    except Exception as e:
        print(f"🚨 Pipeline failed: {e}")
