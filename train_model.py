# ==============================
# 📦 IMPORTS
# ==============================
import pandas as pd
import numpy as np
import boto3
import io
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sklearn.model_selection import train_test_split
from botocore.exceptions import ClientError
import time

# ==============================
# ⚙️ CONFIGURATION
# ==============================
s3_bucket = "healthcarepatientrisk"
s3_key = "diabetic_data.csv"
timestamp = int(time.time())
prefix = f"diabetes-readmission/data-{timestamp}"
endpoint_name = "healthcarepredict"

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker")

# ==============================
# 🧹 STEP 1: LOAD AND PREPARE DATA
# ==============================
print("📥 Loading dataset...")
obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

# 🆕 Include new features
feature_cols = [
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "age",
    "number_diagnoses"
]

df = df[feature_cols + ["readmitted"]]

# Encode target
df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# Clean and encode features
df = df.fillna(0)

# Convert age to numeric if categorical
if df["age"].dtype == object:
    df["age"] = df["age"].replace({
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
        "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    })

# Split train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save CSVs (label first)
for name, dataset in zip(["train", "test"], [train_df, test_df]):
    label = dataset.pop("readmitted")
    dataset.insert(0, "readmitted", label)
    dataset.to_csv(f"{name}.csv", index=False, header=False)

print("✅ Data ready with new features!")

# ==============================
# ☁️ STEP 2: UPLOAD TO S3
# ==============================
train_path = session.upload_data("train.csv", bucket=s3_bucket, key_prefix=f"{prefix}/train")
test_path = session.upload_data("test.csv", bucket=s3_bucket, key_prefix=f"{prefix}/validation")

print("✅ Uploaded to S3:")
print("Train:", train_path)
print("Validation:", test_path)

# ==============================
# 🧠 STEP 3: DEFINE ESTIMATOR
# ==============================
container = image_uris.retrieve("xgboost", region=region, version="1.7-1")

xgb_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{s3_bucket}/{prefix}/output",
    base_job_name="diabetes-xgb",
    sagemaker_session=session,
)

xgb_estimator.set_hyperparameters(
    objective="binary:logistic",
    eval_metric="auc",
    num_round=100
)

# ==============================
# 🔧 STEP 4: DEFINE TUNING PARAMETERS
# ==============================
hyperparameter_ranges = {
    "eta": ContinuousParameter(0.01, 0.3),
    "max_depth": IntegerParameter(3, 10),
    "min_child_weight": IntegerParameter(1, 10),
    "subsample": ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0)
}

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[{"Name": "validation:auc", "Regex": "validation-auc=([0-9\\.]+)"}],
    max_jobs=3,
    max_parallel_jobs=1,
    objective_type="Maximize",
    strategy="Bayesian",
    early_stopping_type="Auto",
    random_seed=42
)

# ==============================
# 🚀 STEP 5: LAUNCH TUNING JOB
# ==============================
print("🚀 Launching tuning job...")
tuner.fit(
    inputs={
        "train": sagemaker.inputs.TrainingInput(train_path, content_type="text/csv"),
        "validation": sagemaker.inputs.TrainingInput(test_path, content_type="text/csv")
    },
    wait=True
)
print("✅ Tuning complete!")

# ==============================
# 🏆 STEP 6: GET BEST MODEL
# ==============================
best_job = tuner.best_training_job()
info = sm_client.describe_training_job(TrainingJobName=best_job)
model_artifact = info["ModelArtifacts"]["S3ModelArtifacts"]
print("🏆 Best job:", best_job)
print("📦 Model artifact:", model_artifact)

# ==============================
# 🧩 STEP 7: DEPLOY MODEL
# ==============================
def endpoint_exists(name):
    try:
        sm_client.describe_endpoint(EndpointName=name)
        return True
    except ClientError:
        return False

if endpoint_exists(endpoint_name):
    print(f"⚠️ Endpoint '{endpoint_name}' exists — deleting...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("⏳ Waiting for deletion to complete...")
    # Wait manually until endpoint is gone
    while True:
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
            time.sleep(15)
        except ClientError:
            break
    print("✅ Old endpoint deleted.")

# Deploy new best model
from sagemaker.model import Model
best_model = Model(
    image_uri=container,
    model_data=model_artifact,
    role=role,
    sagemaker_session=session
)

print(f"🚀 Deploying best model to endpoint '{endpoint_name}'...")
best_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)
print("✅ Endpoint deployed!")

# ==============================
# 🔮 STEP 8: CREATE PREDICTOR & TEST
# ==============================
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Take a few test samples
sample = test_df.iloc[:3, 1:]  # exclude label
payload = sample.to_csv(index=False, header=False)
print("⚙️ Making prediction...")
result = predictor.predict(payload)
print("✅ Predictions:", result)

# Example for new patient
new_patient = pd.DataFrame([{
    "time_in_hospital": 4,
    "num_medications": 10,
    "num_lab_procedures": 35,
    "num_procedures": 1,
    "age": 65,
    "number_diagnoses": 6
}])

payload = new_patient.to_csv(index=False, header=False)
prediction = predictor.predict(payload)
print("📊 New patient prediction:", prediction)

# Optional cleanup
# predictor.delete_endpoint()
# print("🧹 Endpoint deleted.")
