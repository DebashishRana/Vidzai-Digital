import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

model = None
vectorizer = None
MODEL_AVAILABLE = False


def _ensure_models_loaded() -> bool:
    global model, vectorizer, MODEL_AVAILABLE
    if MODEL_AVAILABLE:
        return True
    
    try:
        if model is None:
            model_path = MODEL_DIR / "text_field_classifier.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
        
        if vectorizer is None:
            primary = MODEL_DIR / "vectorizer.joblib"
            legacy = MODEL_DIR / "tfidf_vectorizer.joblib"
            if primary.exists():
                vectorizer = joblib.load(primary)
            elif legacy.exists():
                vectorizer = joblib.load(legacy)
        
        if model and vectorizer:
            MODEL_AVAILABLE = True
            print("✅ Text classifier models loaded")
            return True
        else:
            print("⚠️  Text classifier models not found - using mock classification")
            return False
    except Exception as e:
        print(f"⚠️  Failed to load classifier models: {str(e)}")
        return False


def classify_fields(texts: list[str]) -> list[dict]:
    """Classify extracted text lines into field types (TOTAL, DATE, TAX, AMOUNT, OTHER)."""
    if not texts:
        return []
    
    # Try to load models
    if not _ensure_models_loaded():
        # Return mock classification for demo mode
        return generate_mock_classification(texts)
    
    try:
        X = vectorizer.transform(texts)
        predictions = model.predict(X)
        return [{"text": t, "field": p} for t, p in zip(texts, predictions)]
    except Exception as e:
        print(f"⚠️  Classification error: {str(e)} - using mock data")
        return generate_mock_classification(texts)


def generate_mock_classification(texts: list[str]) -> list[dict]:
    """Generate mock field classification for demo/testing purposes."""
    field_types = ["DATE", "NAME", "AMOUNT", "ADDRESS", "ID", "OTHER"]
    results = []
    
    for i, text in enumerate(texts):
        field_type = field_types[i % len(field_types)]
        results.append({
            "text": text,
            "field": field_type
        })
    
    return results
