# eKYC Cloud Run Deployment Script (PowerShell)
# For GCP Project: vidzai (asia-south2:m10)

# Fix PATH to include Google Cloud SDK
$env:PATH = "C:\Users\simon\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin;" + $env:PATH

$PROJECT_ID = "vidzai"
$CLOUD_SQL_INSTANCE = "vidzai:asia-south2:m10"
$REGION = "asia-south2"
$IMAGE_NAME = "ekyc-backend"
$IMAGE_TAG = "gcr.io/vidzai/ekyc-backend:latest"
$CONTAINER_PORT = 8080

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "eKYC Backend Deployment to Cloud Run" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID" -ForegroundColor Yellow
Write-Host "Region: $REGION" -ForegroundColor Yellow
Write-Host "Cloud SQL: $CLOUD_SQL_INSTANCE" -ForegroundColor Yellow
Write-Host ""

# Step 1: Authenticate with GCP
Write-Host "1/4 - Authenticating with GCP..." -ForegroundColor Blue
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
gcloud auth configure-docker
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Step 2: Build Docker image
Write-Host ""
Write-Host "2/4 - Building Docker image..." -ForegroundColor Blue
docker build -t $IMAGE_TAG .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "OK: Docker build successful" -ForegroundColor Green

# Step 3: Push to Google Container Registry
Write-Host ""
Write-Host "3/4 - Pushing image to Container Registry..." -ForegroundColor Blue
docker push $IMAGE_TAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed" -ForegroundColor Red
    exit 1
}
Write-Host "OK: Image pushed successfully" -ForegroundColor Green

# Step 4: Deploy to Cloud Run
Write-Host ""
Write-Host "4/4 - Deploying to Cloud Run..." -ForegroundColor Blue

$DATABASE_URL = "mysql+pymysql://KYC_USER:Lord@8102@/ekyc?unix_socket=/cloudsql/$CLOUD_SQL_INSTANCE"

gcloud run deploy $IMAGE_NAME --image $IMAGE_TAG --platform managed --region $REGION --add-cloudsql-instances $CLOUD_SQL_INSTANCE --memory 2Gi --cpu 2 --timeout 300 --set-env-vars "DATABASE_URL=$DATABASE_URL,ENVIRONMENT=production,GCP_PROJECT_ID=$PROJECT_ID" --allow-unauthenticated --max-instances 100

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS: Deployment completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "eKYC Backend is LIVE on Cloud Run!" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $SERVICE_URL = (gcloud run services describe $IMAGE_NAME --platform managed --region $REGION --format='value(status.url)')
    Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Test backend: curl $SERVICE_URL/health" -ForegroundColor White
    Write-Host "2. Check stats: curl $SERVICE_URL/stats" -ForegroundColor White
    Write-Host "3. Update frontend URLs to: $SERVICE_URL" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "ERROR: Deployment failed" -ForegroundColor Red
    exit 1
}
