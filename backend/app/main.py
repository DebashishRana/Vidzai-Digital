from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import document, ocr, verification, dashboard, address, verification_logs
from app.address_detection import initialize_detector

app = FastAPI(
    title="KYC Verification Setup G13-M10",
    description="AI-Powered KYC Verification System with Document Detection, OCR, and Face Verification",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since you are opening index.html this will prevent CORS blocks
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document.router, prefix="/api/v1")
app.include_router(ocr.router, prefix="/api/v1")
app.include_router(verification.router, prefix="/api/v1")
app.include_router(address.router, prefix="/api/v1")
app.include_router(verification_logs.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1/dashboard")


@app.on_event("startup")
async def startup_event():
    """Load detector on startup but do not fail service if model init fails."""
    try:
        print("🚀 Initializing address detector...")
        initialize_detector()
        print("✅ Address detector initialized")
    except Exception as e:
        print(f"⚠️ Address detector init failed, continuing in degraded mode: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}
