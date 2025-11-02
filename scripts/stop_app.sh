#!/bin/bash
set -e

APP_NAME="patient-risk"

if [ "$(docker ps -q -f name=$APP_NAME)" ]; then
    echo "🛑 Stopping container $APP_NAME..."
    docker stop $APP_NAME
    docker rm $APP_NAME
    echo "✅ Container stopped and removed."
else
    echo "ℹ️ No container named $APP_NAME is running."
fi
