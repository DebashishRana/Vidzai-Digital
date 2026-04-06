# eKYC Backend - Database & Deployment Setup Guide

## QUICK START (2-3 Hours)

### STEP 1: Local MySQL Setup (30 mins)

#### 1.1 Install MySQL (Windows)
```bash
# Option A: Download MySQL from https://dev.mysql.com/downloads/mysql/
# Option B: Use WSL
wsl
sudo apt-get install mysql-server
sudo service mysql start

# Verify installation
mysql --version
```

#### 1.2 Create Database & User
```bash
mysql -u root

# Paste this SQL:
```
```sql
CREATE DATABASE ekyc;
CREATE USER 'kyc_user'@'localhost' IDENTIFIED BY 'kyc_password123';
GRANT ALL PRIVILEGES ON ekyc.* TO 'kyc_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```
```

#### 1.3 Initialize Tables
```bash
cd d:\Python\Infosys\main\backend
mysql -u kyc_user -p kyc_password123 ekyc < init_db.sql

# Verify tables created
mysql -u kyc_user -p kyc_password123 -e "USE ekyc; SHOW TABLES;"
```

---

### STEP 2: Update Backend Configuration (15 mins)

#### 2.1 Install Python Dependencies
```bash
cd d:\Python\Infosys\main\backend

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sqlalchemy; print('✅ SQLAlchemy installed')"
python -c "import pymysql; print('✅ PyMySQL installed')"
python -c "import dotenv; print('✅ python-dotenv installed')"
```

#### 2.2 Verify .env File
The `.env` file should already exist with:
```env
DATABASE_URL=mysql+pymysql://kyc_user:kyc_password123@localhost:3306/ekyc
ENVIRONMENT=development
```

#### 2.3 Test Local Backend
```bash
cd d:\Python\Infosys\main\backend

# Activate virtual environment if needed
.venv\Scripts\activate

# Start backend
python -m uvicorn main:app --reload --port 8000

# In another terminal, test:
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

Expected output:
```json
{"status": "healthy", "service": "eKYC Address Detector", "version": "1.0.0"}
{"total_verifications": 0, "verified": 0, "flagged": 0, "pending": 0, "alerts": 0}
```

---

### STEP 3: Build & Test Docker Locally (30 mins)

#### 3.1 Build Docker Image
```bash
cd d:\Python\Infosys\main\backend

# Build the image
docker build -t ekyc-backend:latest .

# Verify build
docker images | grep ekyc-backend
```

#### 3.2 Run Container with Local MySQL
```bash
# Run container connected to local MySQL
docker run \
  --name ekyc-backend \
  -p 8000:8080 \
  -e DATABASE_URL="mysql+pymysql://kyc_user:kyc_password123@host.docker.internal:3306/ekyc" \
  -e ENVIRONMENT=production \
  ekyc-backend:latest

# Test in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Stop container
docker stop ekyc-backend
```

---

### STEP 4: Google Cloud Setup (45 mins)

#### 4.1 Create GCP Project
```bash
# Create new project
gcloud projects create ekyc-app --name="eKYC Application"

# Set as active project
gcloud config set project ekyc-app

# Note your PROJECT_ID displayed
```

#### 4.2 Enable Required APIs
```bash
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com
```

#### 4.3 Create Cloud SQL Instance
```bash
# Create MySQL 8.0 instance (db-f1-micro is free tier eligible)
gcloud sql instances create ekyc-db \
    --database-version=MYSQL_8_0 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --root-password=UpdateMeWithSecurePassword123!

# This takes ~5 minutes...
```

#### 4.4 Create Database User & Database
```bash
# Create user
gcloud sql users create kyc_user \
    --instance=ekyc-db \
    --password=kyc_password123

# Create database
gcloud sql databases create ekyc \
    --instance=ekyc-db
```

#### 4.5 Initialize Database Tables
```bash
# Upload init script
gcloud sql import sql ekyc-db gs://bucket-name/init_db.sql \
    --database=ekyc

# OR manually run SQL
gcloud sql connect ekyc-db --user=root
# Paste content of init_db.sql
```

---

### STEP 5: Deploy to Cloud Run (30 mins)

#### 5.1 Authenticate with GCP
```bash
gcloud auth configure-docker

# Verify authentication
docker ps  # Should not ask for credentials
```

#### 5.2 Get Cloud SQL Connection Name
```bash
gcloud sql instances describe ekyc-db --format='get(connectionName)'

# Output will be: PROJECT_ID:us-central1:ekyc-db
# Copy this value
```

#### 5.3 Build & Push Docker Image
```bash
cd d:\Python\Infosys\main\backend

# Set variables
$PROJECT_ID = "your-project-id"
$REGION = "us-central1"
$IMAGE_NAME = "ekyc-backend"

# Build image
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

# Verify push
gcloud container images list --repository=gcr.io/$PROJECT_ID
```

#### 5.4 Deploy to Cloud Run
```bash
$PROJECT_ID = "your-project-id"
$CLOUD_SQL_INSTANCE = "PROJECT_ID:us-central1:ekyc-db"
$DATABASE_URL = "mysql+pymysql://kyc_user:kyc_password123@/ekyc?unix_socket=/cloudsql/$CLOUD_SQL_INSTANCE"

gcloud run deploy ekyc-backend `
    --image gcr.io/$PROJECT_ID/ekyc-backend:latest `
    --platform managed `
    --region us-central1 `
    --add-cloudsql-instances $CLOUD_SQL_INSTANCE `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --set-env-vars "DATABASE_URL=$DATABASE_URL,ENVIRONMENT=production" `
    --allow-unauthenticated

# Wait for deployment...
```

#### 5.5 Get Cloud Run URL
```bash
gcloud run services describe ekyc-backend \
    --platform managed \
    --region us-central1 \
    --format='value(status.url)'

# Output: https://ekyc-backend-xxxxx.run.app
```

---

### STEP 6: Verify Deployment (10 mins)

```bash
# Set the Cloud Run URL
$BACKEND_URL = "https://ekyc-backend-xxxxx.run.app"

# Test health endpoint
curl $BACKEND_URL/health

# Test stats endpoint (should be empty initially)
curl $BACKEND_URL/stats

# Test verification endpoint (upload image)
# Use Postman or curl with image file
```

---

## Environment Variables for Different Stages

### Development (Local)
```bash
DATABASE_URL=mysql+pymysql://kyc_user:kyc_password123@localhost:3306/ekyc
ENVIRONMENT=development
```

### Production (Cloud Run with Cloud SQL)
```bash
DATABASE_URL=mysql+pymysql://kyc_user:kyc_password123@/ekyc?unix_socket=/cloudsql/PROJECT_ID:us-central1:ekyc-db
ENVIRONMENT=production
```

### Cloud Run with Socket Connection
```bash
gcloud run deploy ekyc-backend \
    --add-cloudsql-instances PROJECT_ID:us-central1:ekyc-db \
    --set-env-vars "DATABASE_URL=mysql+pymysql://kyc_user:kyc_password123@/ekyc?unix_socket=/cloudsql/PROJECT_ID:us-central1:ekyc-db"
```

---

## Troubleshooting

### MySQL Connection Error
```
Error: Can't connect to MySQL server
```
**Solution:**
- Check MySQL service is running: `sudo service mysql status`
- Verify credentials in .env file
- Test connection: `mysql -u kyc_user -p kyc_password123 -e "SELECT 1"`

### Docker Build Fails
```
Error: failed to solve with frontend dockerfile.v0
```
**Solution:**
- Check Docker daemon is running: `docker ps`
- Clear cache: `docker system prune`
- Rebuild: `docker build --no-cache -t ekyc-backend:latest .`

### Cloud Run Deployment Timeout
```
Error: Cloud Run deployment failed
```
**Solution:**
- Check Cloud SQL instance is ready: `gcloud sql instances describe ekyc-db`
- Verify network connectivity: `gcloud sql instances patch ekyc-db --assign-ip`
- Check logs: `gcloud run logs read ekyc-backend --limit 50`

### Image Not Found in Registry
```
Error: image not found
```
**Solution:**
- Verify push was successful: `gcloud container images list`
- Check image tag: `docker images | grep ekyc`
- Re-tag and push: `docker tag ekyc-backend:latest gcr.io/PROJECT_ID/ekyc-backend:latest`

---

## Files Created/Modified

✅ **New Files:**
- `.env` - Environment configuration
- `app/models.py` - SQLAlchemy ORM models
- `init_db.sql` - Database initialization script
- `Dockerfile` - Container image definition
- `deploy.sh` - Automated deployment script
- `cloudbuild.yaml` - Cloud Build configuration

✅ **Modified Files:**
- `app/database.py` - Updated to use .env and real MySQL
- `app/crud.py` - Rewritten for new models
- `app/routers/verification.py` - Updated to save to database
- `app/routers/dashboard.py` - Updated to fetch from database
- `requirements.txt` - Added sqlalchemy, pymysql, python-dotenv

---

## Next Steps After Deployment

1. **Update Frontend URLs**
   - Replace `localhost:8000` with your Cloud Run URL
   - Update all `.html` files in frontend

2. **Deploy Frontend to Cloud Storage**
   ```bash
   gsutil mb gs://project-id-ekyc-frontend/
   gsutil -m cp -r frontend/ai\ powered\ ekyc/* gs://project-id-ekyc-frontend/
   ```

3. **Setup Monitoring**
   - Enable Cloud Monitoring for performance tracking
   - Setup error alerting

4. **Add Authentication** (After Demo)
   - Implement JWT tokens
   - Add rate limiting

---

## Support Commands

```bash
# View logs
gcloud run logs read ekyc-backend --limit 50 --follow

# Check Cloud SQL status
gcloud sql instances describe ekyc-db --format=json

# Connect to Cloud SQL
gcloud sql connect ekyc-db --user=root

# Redeploy latest code
gcloud run deploy ekyc-backend \
    --image gcr.io/PROJECT_ID/ekyc-backend:latest \
    --platform managed \
    --region us-central1

# View deployment history
gcloud run revisions list --service=ekyc-backend
```

---

**Estimated Total Time: 2.5 - 3 hours**
