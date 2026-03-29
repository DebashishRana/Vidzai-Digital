#!/bin/bash

# EKYC Cloud Run Deployment Script
# For GCP Project: vidzai (asia-south2:m10)

PROJECT_ID="vidzai"
CLOUD_SQL_INSTANCE="vidzai:asia-south2:m10"
REGION="asia-south2"
IMAGE_NAME="ekyc-backend"
CONTAINER_PORT=8080

echo "========================================="
echo "eKYC Backend Deployment to Cloud Run"
echo "========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Cloud SQL: $CLOUD_SQL_INSTANCE"
echo ""

# Step 1: Authenticate with GCP
echo "🔐 Authenticating with GCP..."
gcloud auth configure-docker
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Step 2: Build Docker image
echo ""
echo "🔨 Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
    --build-arg DATABASE_URL="mysql+pymysql://KYC_USER:Lord@8102@/ekyc?unix_socket=/cloudsql/$CLOUD_SQL_INSTANCE" \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✅ Docker build successful"

# Step 3: Push to Google Container Registry
echo ""
echo "📤 Pushing image to Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi
echo "✅ Image pushed successfully"

# Step 4: Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."

gcloud run deploy $IMAGE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --add-cloudsql-instances $CLOUD_SQL_INSTANCE \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars "DATABASE_URL=mysql+pymysql://KYC_USER:Lord@8102@/ekyc?unix_socket=/cloudsql/$CLOUD_SQL_INSTANCE,ENVIRONMENT=production,GCP_PROJECT_ID=$PROJECT_ID" \
    --allow-unauthenticated \
    --max-instances 100

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "========================================="
    echo "Your eKYC Backend is now LIVE!"
    echo "========================================="
    echo ""
    
    SERVICE_URL=$(gcloud run services describe $IMAGE_NAME --platform managed --region $REGION --format='value(status.url)')
    echo "🌐 Service URL: $SERVICE_URL"
    echo ""
    echo "📝 Update your frontend with:"
    echo "   const API_BASE = '$SERVICE_URL'"
    echo ""
    echo "✅ Test endpoints:"
    echo "   curl $SERVICE_URL/health"
    echo "   curl $SERVICE_URL/stats"
    echo ""
else
    echo "❌ Deployment failed"
    exit 1
fi
