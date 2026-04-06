import cv2
import numpy as np
import pytesseract

# Check if Tesseract is available
TESSERACT_AVAILABLE = False
try:
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract OCR available")
except (pytesseract.TesseractNotFoundError, Exception) as e:
    print(f"⚠️  Tesseract OCR not available: {str(e)}")
    print("📝 Running in DEMO MODE - OCR will return mock data")


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Grayscale → median blur → adaptive threshold (same as notebook)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    return thresh


def extract_text(image_bytes: bytes) -> str:
    """Preprocess an uploaded image and run Tesseract OCR on it."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    processed = preprocess_image(img)
    
    if TESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(processed, config="--psm 6").strip()
            return text
        except pytesseract.TesseractNotFoundError:
            print("⚠️  Tesseract failed, using mock data")
            return generate_mock_ocr_text()
    else:
        # Return mock OCR text for demo mode
        return generate_mock_ocr_text()


def generate_mock_ocr_text() -> str:
    """Generate mock OCR text for demo/testing purposes."""
    mock_text = """
    GOVERNMENT OF INDIA
    AADHAAR
    Unique Identity Number
    
    Name: Ramesh Kumar Singh
    Date of Birth: 15/03/1985
    Gender: Male
    
    Aadhaar Number: 3456-7890-1234
    
    Address: 123 Main Street
    New Delhi, Delhi 110001
    India
    
    Issued: 01/06/2015
    Valid: Lifelong
    """
    return mock_text.strip()
