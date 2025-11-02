#!/bin/bash
set -e

APP_NAME="patient-risk"
IMAGE_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/patient-risk-app:latest"
PORT=8501

echo "🚀 Starting Streamlit app..."

# Stop and remove existing container if running
if [ "$(docker ps -q -f name=$APP_NAME)" ]; then
    echo "🧹 Stopping existing container..."
    docker stop $APP_NAME
    docker rm $APP_NAME
fi

# Pull latest image
echo "⬇️ Pulling latest image from ECR..."
docker pull $IMAGE_NAME

# Run new container
echo "🧱 Running container on port $PORT..."
docker run -d -p $PORT:8501 --name $APP_NAME $IMAGE_NAME

echo "✅ Streamlit app is up and running!"
