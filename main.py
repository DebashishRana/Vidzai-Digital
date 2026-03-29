"""
FastAPI Backend for eKYC - Address Detection Endpoint
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
import shutil

from app.address_detection import get_detector, initialize_detector
from app.routers import verification, document, dashboard

app = FastAPI(title="eKYC Address Detector", version="1.0.0")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(verification.router)
app.include_router(document.router)
app.include_router(dashboard.router)

@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    from app.database import DB_ENABLED
    
    print("🚀 Initializing models...")
    initialize_detector()
    print("✅ Server ready")
    
    if DB_ENABLED:
        print("✅ Database connected")
    else:
        print("⚠️  Running in DEMO MODE (in-memory storage)")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "eKYC Address Detector",
        "models_loaded": True
    }

@app.post("/detect/address")
async def detect_address(file: UploadFile = File(...)):
    """
    Detect and extract address from invoice image
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        JSON with extracted address or error message
    """
    try:
        # Save temp file
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / file.filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with detector
        detector = get_detector()
        result = detector.detect_and_extract(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        if result is None:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": "No address box detected in image",
                    "image": file.filename
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "image": result["image"],
                "address": result["address"],
                "confidence": result["confidence"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/batch/address")
async def batch_detect_addresses(files: list[UploadFile] = File(...)):
    """
    Batch process multiple invoice images for address extraction
    
    Args:
        files: List of image files
        
    Returns:
        List of results for each image
    """
    results = []
    detector = get_detector()
    
    for i, file in enumerate(files):
        try:
            temp_dir = tempfile.gettempdir()
            temp_path = Path(temp_dir) / file.filename
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = detector.detect_and_extract(str(temp_path))
            temp_path.unlink()
            
            if result:
                results.append({
                    "index": i,
                    "success": True,
                    **result
                })
            else:
                results.append({
                    "index": i,
                    "success": False,
                    "image": file.filename,
                    "reason": "No address detected"
                })
                
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "image": file.filename,
                "error": str(e)
            })
    
    return {
        "total_processed": len(files),
        "successful": len([r for r in results if r.get("success")]),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
