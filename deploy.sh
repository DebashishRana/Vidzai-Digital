#!/bin/bash

# eKYC Backend Deployment Script for Google Cloud Run
# Usage: bash deploy.sh <PROJECT_ID> <DATABASE_URL>

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy.sh <GCP_PROJECT_ID> [DATABASE_URL]"
    echo ""
    echo "Example:"
    echo "  bash deploy.sh my-project-123"
    echo "  bash deploy.sh my-project-123 'mysql+pymysql://user:pass@cloudsql-ip/ekyc'"
    exit 1
fi

PROJECT_ID=$1
DATABASE_URL=${2:-"mysql+pymysql://kyc_user:kyc_password123@localhost:3306/ekyc"}

echo "========================================="
echo "eKYC Backend Deployment to Cloud Run"
echo "========================================="
echo "Project ID: $PROJECT_ID"
echo "Database URL: $DATABASE_URL"
echo ""

# Set project
gcloud config set project $PROJECT_ID
echo "✅ Project set to $PROJECT_ID"
echo ""

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/ekyc-backend:latest .
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✅ Docker build successful"
echo ""

# Push image to Google Container Registry
echo "📤 Pushing image to Container Registry..."
docker push gcr.io/$PROJECT_ID/ekyc-backend:latest
if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi
echo "✅ Image pushed successfully"
echo ""

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ekyc-backend \
    --image gcr.io/$PROJECT_ID/ekyc-backend:latest \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars "DATABASE_URL=$DATABASE_URL,ENVIRONMENT=production" \
    --allow-unauthenticated

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "========================================="
    echo "Your eKYC Backend is now live!"
    echo "========================================="
    gcloud run services describe ekyc-backend --platform managed --region us-central1 --format='value(status.url)'
else
    echo "❌ Deployment failed"
    exit 1
fi
