# QUICK START: Deploy eKYC to Cloud Run (vidzai:asia-south2:m10)

## YOUR GCP DETAILS (Saved ✅)
```
Project ID:      vidzai
Region:          asia-south2
Cloud SQL:       vidzai:asia-south2:m10
Public IP:       34.131.169.73
Database:        ekyc
Username:        KYC_USER
Password:        Lord@8102
```

---

## STEP 1: Local Testing with Real Cloud SQL (10 mins)

### 1.1 Update .env (Already Done ✅)
File: `d:\Python\Infosys\main\backend\.env`

Current setting uses public IP:
```
DATABASE_URL=mysql+pymysql://KYC_USER:Lord@8102@34.131.169.73:3306/ekyc
```

### 1.2 Test Backend Locally
```bash
cd d:\Python\Infosys\main\backend

# Install/Update requirements
pip install -r requirements.txt

# Start backend
python -m uvicorn main:app --reload --port 8000

# In another terminal, test connection:
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

Expected output:
```json
{"status": "healthy", "service": "eKYC Address Detector"}
{"total_verifications": 0, "verified": 0, "flagged": 0, "pending": 0, "alerts": 0}
```

---

## STEP 2: Build Docker Image (15 mins)

### 2.1 Build
```bash
cd d:\Python\Infosys\main\backend

docker build -t gcr.io/vidzai/ekyc-backend:latest .
```

### 2.2 Verify Build
```bash
docker images | grep ekyc-backend
```

Should show: `gcr.io/vidzai/ekyc-backend   latest`

---

## STEP 3: Deploy to Cloud Run (30 mins)

### 3.1 Use PowerShell Script (EASIEST - Windows)
```powershell
cd d:\Python\Infosys\main\backend

# Run deployment script
.\deploy-vidzai.ps1
```

**OR** Use Bash script:
```bash
bash deploy-vidzai.sh
```

### 3.2 Manual Deployment (If scripts don't work)

```powershell
$PROJECT_ID = "vidzai"
$CLOUD_SQL = "vidzai:asia-south2:m10"

# Authenticate
gcloud auth configure-docker
gcloud config set project vidzai
gcloud config set compute/region asia-south2

# Build
docker build -t gcr.io/vidzai/ekyc-backend:latest .

# Push
docker push gcr.io/vidzai/ekyc-backend:latest

# Deploy
gcloud run deploy ekyc-backend `
    --image gcr.io/vidzai/ekyc-backend:latest `
    --platform managed `
    --region asia-south2 `
    --add-cloudsql-instances vidzai:asia-south2:m10 `
    --memory 2Gi `
    --cpu 2 `
    --set-env-vars "DATABASE_URL=mysql+pymysql://KYC_USER:Lord@8102@/ekyc?unix_socket=/cloudsql/vidzai:asia-south2:m10,ENVIRONMENT=production" `
    --allow-unauthenticated
```

---

## STEP 4: Verify Deployment (5 mins)

### 4.1 Get Service URL
```bash
gcloud run services describe ekyc-backend `
    --platform managed `
    --region asia-south2 `
    --format='value(status.url)'
```

Output: `https://ekyc-backend-xxxxx.run.app`

### 4.2 Test Cloud Service
```bash
curl https://ekyc-backend-xxxxx.run.app/health
curl https://ekyc-backend-xxxxx.run.app/stats
```

### 4.3 Check Logs
```bash
gcloud run logs read ekyc-backend --limit 50 --follow
```

---

## STEP 5: Update Frontend (5 mins)

Replace all backend URLs in frontend files:

### Files to update:
- `frontend/ai powered ekyc/ai powered ekyc/dashboard.html`
- `frontend/ai powered ekyc/ai powered ekyc/verification_flow.html`
- `frontend/ai powered ekyc/ai powered ekyc/verifications.html`
- `frontend/ai powered ekyc/ai powered ekyc/alerts.html`

### Find & Replace:
```
OLD: http://localhost:8000
NEW: https://ekyc-backend-xxxxx.run.app
```

Example for dashboard.html:
```javascript
// OLD:
const response = await fetch('http://localhost:8000/stats');

// NEW:
const response = await fetch('https://ekyc-backend-xxxxx.run.app/stats');
```

---

## COMPLETE DEPLOYMENT CHECKLIST

### Prerequisites
- [x] GCP Account & vidzai project created
- [x] Cloud SQL m10 instance ready
- [x] Database 'ekyc' created
- [x] User 'KYC_USER' with password 'Lord@8102' created
- [x] Docker installed locally
- [x] gcloud CLI installed
- [x] Python 3.11+ installed

### Local Testing
- [ ] Backend starts with `python -m uvicorn main:app --reload`
- [ ] Can reach Cloud SQL: `curl http://localhost:8000/stats` shows 0 verifications
- [ ] Frontend connects to localhost:8000

### Docker Build
- [ ] `docker build -t gcr.io/vidzai/ekyc-backend:latest .` succeeds
- [ ] `docker images` shows the built image

### Cloud Run Deployment
- [ ] `gcloud auth configure-docker` completes
- [ ] `docker push gcr.io/vidzai/ekyc-backend:latest` succeeds
- [ ] `gcloud run deploy` completes without errors
- [ ] Got service URL from `gcloud run services describe`

### Verification
- [ ] Cloud service health: `curl <URL>/health` returns 200
- [ ] Stats endpoint: `curl <URL>/stats` returns JSON
- [ ] Logs show no errors: `gcloud run logs read ekyc-backend`

### Frontend Integration
- [ ] Updated all .html files with new API URL
- [ ] Frontend loads dashboard
- [ ] Can upload image and see it process
- [ ] Data appears in Cloud SQL

---

## TROUBLESHOOTING

### Docker Build Fails
```bash
# Clear cache and retry
docker system prune -a
docker build --no-cache -t gcr.io/vidzai/ekyc-backend:latest .
```

### Push Fails
```bash
# Reconfigure Docker auth
gcloud auth configure-docker
docker push gcr.io/vidzai/ekyc-backend:latest
```

### Cloud SQL Connection Error
```
Error: Can't connect to MySQL server
```

**Solution:**
1. Check m10 instance is running: `gcloud sql instances describe m10`
2. Verify firewall allows your IP
3. Test direct connection: `mysql -h 34.131.169.73 -u KYC_USER -pLord@8102 -e "SELECT 1"`

### Deployment Timeout
```bash
# Check logs for issues
gcloud run logs read ekyc-backend --limit 100 --follow

# If still erroring, rollback
gcloud run revisions list --service=ekyc-backend
gcloud run deploy ekyc-backend --revision=<previous-revision-name>
```

### Tables Not Created
```bash
# Manually initialize (SQLAlchemy should do this automatically)
gcloud sql connect m10 --user=root
# Paste init_db.sql content
```

---

## TOTAL TIME ESTIMATE

| Step | Time |
|------|------|
| Local Testing | 15 mins |
| Docker Build | 10 mins |
| Cloud Run Deploy | 20 mins |
| Verification | 5 mins |
| Frontend Update | 5 mins |
| **TOTAL** | **~55 mins** |

---

## IMPORTANT NOTES

✅ **What's Automated:**
- Database tables created automatically on first run (SQLAlchemy)
- Connection pooling handled automatically
- Fallback to in-memory if Cloud SQL fails
- Health checks built into Docker image

⚠️ **Manual Steps:**
1. Run deployment script
2. Update frontend URLs
3. Test end-to-end

---

## COMMANDS SUMMARY

```bash
# Test locally
python -m uvicorn main:app --reload

# Build Docker
docker build -t gcr.io/vidzai/ekyc-backend:latest .

# Deploy (choose one)
.\deploy-vidzai.ps1                    # PowerShell (Windows)
bash deploy-vidzai.sh                  # Bash (Linux/Mac)

# Verify
gcloud run services describe ekyc-backend --platform managed --region asia-south2 --format='value(status.url)'
curl https://ekyc-backend-xxxxx.run.app/health

# Logs
gcloud run logs read ekyc-backend --limit 50 --follow
```

---

## NEXT AFTER DEPLOYMENT

1. ✅ Upload test image - should save to Cloud SQL
2. ✅ Check dashboard - should show stats
3. ✅ Test verifications page - should list uploads
4. ✅ Move to production (add authentication, monitoring, etc.)

---

**You're all set! Run the deployment script now! 🚀**
