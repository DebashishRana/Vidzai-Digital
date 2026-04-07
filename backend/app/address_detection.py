"""
Address Detection Module - Integrates YOLO + EasyOCR for address extraction.
"""

from pathlib import Path
import cv2
import numpy as np
import re
from typing import Optional, Dict
import easyocr
from ultralytics import YOLO
import json
import time

# Load models from backend storage
MODELS_DIR = Path(__file__).parent / "models"

_DEBUG_SESSION_ID = "96abd7"
def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]

_DEBUG_LOG_PATH = _workspace_root() / "debug-96abd7.log"

def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

class AddressDetector:
    DISALLOWED_ID_PATTERNS = [
        re.compile(r'\bvid\s*[:\-]?\s*(?:\d\s*){16}\b', re.IGNORECASE),
        re.compile(r'\b(?:aadhaar|aadhar)\s*(?:no|number|num)?\s*[:\-]?\s*(?:\d\s*){12}\b', re.IGNORECASE),
    ]

    STRONG_NEGATIVE_TERMS = {
        "uidai",
        "unique identification authority",
        "government of india",
        "govt of india",
        "authority of india",
        "aadhaar",
        "aadhar",
        "enrolment",
        "enrollment",
        "enrolment no",
        "enrollment no",
        "vid",
        "virtual id",
        "qr code",
        "secure qr",
        "offline xml",
        "authentication",
        "electronically generated",
        "aadhaar is not proof of",
        "aadhar is not proof of",
        "aadhar is",
        "aadhaar is",
        "not proof of",
        "information",
        "notice",
        "सूचना",
    }

    MEDIUM_NEGATIVE_TERMS = {
        "download date",
        "issue date",
        "signature",
        "verified",
        "digitally signed",
        "help@uidai",
        "www.uidai.gov.in",
        "male",
        "female",
        "dob",
        "date of birth",
        "year of birth",
        "yob",
    }

    WEAK_NEGATIVE_TERMS = {
        "india",
        "bharat",
        "government",
        "authority",
        "card",
        "identity",
        "proof",
    }

    POSITIVE_ADDRESS_KEYWORDS = {
        "road", "rd",
        "street", "st",
        "nagar",
        "colony",
        "lane",
        "near",
        "behind",
        "sector",
        "block",
        "apartment",
        "floor",
        "flat",
        "building",
        "house", "h.no", "hno", "house no",
        "plot", "plot no",
        "door no", "d.no",
        "khata", "khasra", "khasra no",
        "mohalla", "moh",
        "village", "vill", "gaon",
        "ward", "ward no",
        "panchayat",
        "tehsil", "taluk", "mandal",
        "district", "dist",
        "post", "post office", "po",
        "sub post office", "spo",
        "head post office", "hpo",
        "pin", "pincode", "pin code",
        "area", "locality",
        "phase",
        "extension", "ext",
        "main", "cross",
        "layout",
        "tower", "complex",
        "residency", "residence",
        "villa", "villas",
        "heights",
        "plaza",
        "mall",
        "market",
        "chowk",
        "landmark",
        "opp", "opposite",
        "beside",
        "nearby",
        "in front of",
        "marg",
        "path",
        "highway", "nh", "sh",
        "bypass"
        "road",
        "ring road",
    }

    DOB_PATTERNS = [
        re.compile(r'date\s*of\s*birth', re.IGNORECASE),
        re.compile(r'year\s*of\s*birth', re.IGNORECASE),
        re.compile(r'\byob\b', re.IGNORECASE),
        re.compile(r'\bd\.\s*o\.\s*b\.\b', re.IGNORECASE),
        re.compile(r'birth\s*dob', re.IGNORECASE),
        re.compile(r'\bdob\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', re.IGNORECASE),
    ]

    def __init__(self):
        print("Loading models...")
        # Load YOLO model used for localizing address text regions.
        try:
            model_path = MODELS_DIR / "yolov8n.pt"
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Warning: YOLO load failed: {str(e)}")
            self.model = None

        print("Initializing EasyOCR...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"Warning: EasyOCR init failed: {str(e)}")
            self.reader = None
        
        print("✅ Models loaded successfully")

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'[^A-Za-z0-9,./#\-\s]', ' ', text or '')
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        escaped = re.escape(term.strip().lower())
        # Use word boundaries for alpha-numeric terms, substring for non-word tokens.
        if re.search(r'[a-z0-9]', term.lower()):
            return re.search(rf'\b{escaped}\b', text.lower()) is not None
        return escaped in text.lower()

    @staticmethod
    def _term_hits(text: str, terms: set[str]) -> int:
        return sum(1 for term in terms if AddressDetector._contains_term(text, term))

    @staticmethod
    def _extract_address_from_lines(lines: list[str]) -> str:
        if not lines:
            _dbg("H5", "backend/app/address_detection.py:_extract_address_from_lines", "No OCR lines to score", {"lines": 0})
            return ""

        address_keywords = {
            "address", "road", "rd", "street", "st", "lane", "ln", "nagar",
            "district", "state", "india", "pin", "pincode", "sector", "colony",
            "village", "plot", "flat", "apt", "apartment", "marg",
            "house", "h.no", "hno", "house no", "plot no", "door no", "d.no",
            "khata", "khasra", "khasra no", "mohalla", "moh", "vill", "gaon",
            "ward", "ward no", "panchayat", "tehsil", "taluk", "mandal", "dist",
            "post", "post office", "po", "sub post office", "spo", "head post office", "hpo",
            "pin code", "area", "locality", "phase", "extension", "ext", "main", "cross",
            "layout", "tower", "complex", "residency", "residence", "villa", "villas",
            "heights", "plaza", "mall", "market", "chowk", "landmark", "opp", "opposite",
            "beside", "nearby", "in front of", "path", "highway", "nh", "sh", "bypass", "ring road"
        }
        non_address_markers = {
            "name", "dob", "birth", "gender", "aadhaar", "aadhar", "government",
            "uidai", "father", "husband"
        }

        scored: list[tuple[float, int, str]] = []
        for idx, raw_line in enumerate(lines):
            line = AddressDetector._clean_text(raw_line)
            if len(line) < 4:
                continue

            if any(pattern.search(line) for pattern in AddressDetector.DOB_PATTERNS):
                continue

            # Reject Aadhaar/VID identity-number lines from address extraction.
            if any(pattern.search(line) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
                continue

            strong_hits = AddressDetector._term_hits(line, AddressDetector.STRONG_NEGATIVE_TERMS)
            if strong_hits > 0:
                continue

            medium_hits = AddressDetector._term_hits(line, AddressDetector.MEDIUM_NEGATIVE_TERMS)
            weak_hits = AddressDetector._term_hits(line, AddressDetector.WEAK_NEGATIVE_TERMS)

            lower = line.lower()
            keyword_hits = sum(1 for kw in address_keywords if re.search(rf'\b{re.escape(kw)}\b', lower))
            non_address_hits = sum(1 for kw in non_address_markers if re.search(rf'\b{re.escape(kw)}\b', lower))
            positive_hits = AddressDetector._term_hits(line, AddressDetector.POSITIVE_ADDRESS_KEYWORDS)
            has_digit = 1 if re.search(r'\d', line) else 0
            has_pin = 2 if re.search(r'\b\d{6}\b', line) else 0

            score = (2 * keyword_hits) + (2 * positive_hits) + has_digit + has_pin - non_address_hits - (2 * medium_hits) - weak_hits
            if score > 0:
                scored.append((score, idx, line))

        if not scored:
            _dbg(
                "H6",
                "backend/app/address_detection.py:_extract_address_from_lines",
                "No candidate lines survived scoring",
                {
                    "total_lines": len(lines),
                    "sample_lines": [AddressDetector._clean_text(l)[:120] for l in lines[:6]],
                },
            )
            return ""

        scored.sort(key=lambda x: (-x[0], x[1]))
        best_score = scored[0][0]
        min_score = max(1.0, best_score * 0.5)

        selected = sorted((idx, line) for score, idx, line in scored if score >= min_score)
        grouped: list[list[str]] = []
        current_group: list[str] = []
        last_idx = None

        for idx, line in selected:
            if last_idx is None or idx - last_idx <= 1:
                current_group.append(line)
            else:
                if current_group:
                    grouped.append(current_group)
                current_group = [line]
            last_idx = idx

        if current_group:
            grouped.append(current_group)

        if not grouped:
            return ""

        candidate = max((" ".join(group) for group in grouped), key=len)
        candidate = AddressDetector._clean_text(candidate)

        _dbg(
            "H7",
            "backend/app/address_detection.py:_extract_address_from_lines",
            "Built address candidate from scored groups",
            {
                "total_lines": len(lines),
                "scored_lines": len(scored),
                "best_score": best_score,
                "candidate_preview": candidate[:200],
            },
        )

        if any(pattern.search(candidate) for pattern in AddressDetector.DOB_PATTERNS):
            return ""

        if any(pattern.search(candidate) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
            return ""

        if AddressDetector._term_hits(candidate, AddressDetector.STRONG_NEGATIVE_TERMS) > 0:
            return ""

        medium_hits = AddressDetector._term_hits(candidate, AddressDetector.MEDIUM_NEGATIVE_TERMS)
        weak_hits = AddressDetector._term_hits(candidate, AddressDetector.WEAK_NEGATIVE_TERMS)
        if medium_hits >= 2 or (medium_hits >= 1 and weak_hits >= 2):
            return ""

        if AddressDetector._term_hits(candidate, AddressDetector.POSITIVE_ADDRESS_KEYWORDS) == 0:
            return ""

        if len(candidate) < 12 or len(candidate.split()) < 3:
            return ""

        return candidate
    
    def detect_and_extract(self, image_path: str) -> Optional[Dict]:
        """
        Detect address region in image and extract text via OCR.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                _dbg("H8", "backend/app/address_detection.py:detect_and_extract", "cv2.imread returned None", {"image_path": str(image_path)})
                return None

            h, w = img.shape[:2]
            crop = img
            yolo_conf = 0.0
            
            # If YOLO loaded successfully, crop the image first
            if self.model:
                try:
                    results = self.model.predict(source=image_path, conf=0.20, verbose=False)
                    if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, "conf") else np.array([])

                        x1 = max(0, int(np.min(boxes[:, 0])))
                        y1 = max(0, int(np.min(boxes[:, 1])))
                        x2 = min(w, int(np.max(boxes[:, 2])))
                        y2 = min(h, int(np.max(boxes[:, 3])))

                        if x2 > x1 and y2 > y1:
                            crop = img[y1:y2, x1:x2]

                        if confs.size > 0:
                            yolo_conf = float(np.mean(confs))
                except Exception as model_exc:
                    print(f"Warning: YOLO inference failed, using full image OCR: {model_exc}")

            if self.reader is None:
                print("Warning: OCR reader unavailable, skipping address extraction")
                _dbg("H9", "backend/app/address_detection.py:detect_and_extract", "EasyOCR reader unavailable", {"has_model": bool(self.model)})
                return None

            # OCR extraction on detected crop, with full-image fallback if needed
            ocr_result = self.reader.readtext(crop, detail=1)
            if not ocr_result and crop is not img:
                ocr_result = self.reader.readtext(img, detail=1)

            lines = []
            confs = []
            for item in ocr_result:
                if len(item) >= 3:
                    lines.append(str(item[1]))
                    try:
                        confs.append(float(item[2]))
                    except Exception:
                        pass

            clean_address = self._extract_address_from_lines(lines)
            if not clean_address:
                _dbg(
                    "H10",
                    "backend/app/address_detection.py:detect_and_extract",
                    "OCR ran but address extraction returned empty",
                    {
                        "ocr_items": len(ocr_result) if ocr_result is not None else None,
                        "lines_count": len(lines),
                        "lines_preview": [self._clean_text(l)[:120] for l in lines[:6]],
                        "yolo_used": bool(self.model),
                        "yolo_conf": yolo_conf,
                    },
                )
                return None

            ocr_conf = float(np.mean(confs)) if confs else 0.0
            confidence = ocr_conf if yolo_conf <= 0 else (0.6 * ocr_conf + 0.4 * yolo_conf)
            confidence = float(max(0.0, min(1.0, confidence)))

            return {
                "image_name": Path(image_path).name,
                "address": clean_address,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None


# Initialize detector globally
detector = None

def initialize_detector():
    global detector
    detector = AddressDetector()

def get_detector():
    global detector
    if detector is None:
        initialize_detector()
    return detector
