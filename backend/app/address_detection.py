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

# Load models from backend storage
MODELS_DIR = Path(__file__).parent / "models"

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
        "bypass",
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
        """Extract address from OCR lines with relaxed requirements for Aadhaar cards."""
        if not lines:
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
            "beside", "nearby", "in front of", "path", "highway", "nh", "sh", "bypass", "ring road",
            # Additional Aadhaar-specific keywords
            "building", "block", "apartment", "premises", "location", "city", "state", "country",
            "floor", "unit", "society", "complex", "towers"
        }
        non_address_markers = {
            "name", "dob", "birth", "gender", "aadhaar", "aadhar", "government",
            "uidai", "father", "husband", "spouse", "mother"
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
            
            # For Aadhaar cards: relax scoring - score based on length + keywords, not strict requirements
            has_comma_or_slash = 1 if re.search(r'[,/]', line) else 0
            line_length_bonus = min(2, len(line) // 20)  # Bonus for longer lines

            score = (2 * keyword_hits) + (2 * positive_hits) + has_digit + has_pin + has_comma_or_slash + line_length_bonus - non_address_hits - (2 * medium_hits) - weak_hits
            if score > 0 or (len(line) > 20 and has_digit and not non_address_hits > 1):
                scored.append((score, idx, line))

        if not scored:
            return ""

        scored.sort(key=lambda x: (-x[0], x[1]))
        best_score = scored[0][0]
        min_score = max(0.5, best_score * 0.4)  # Relaxed minimum score threshold

        selected = sorted((idx, line) for score, idx, line in scored if score >= min_score)
        grouped: list[list[str]] = []
        current_group: list[str] = []
        last_idx = None

        for idx, line in selected:
            if last_idx is None or idx - last_idx <= 2:  # Allow gaps of up to 2 lines
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

        if any(pattern.search(candidate) for pattern in AddressDetector.DOB_PATTERNS):
            return ""

        if any(pattern.search(candidate) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
            return ""

        if AddressDetector._term_hits(candidate, AddressDetector.STRONG_NEGATIVE_TERMS) > 0:
            return ""

        medium_hits = AddressDetector._term_hits(candidate, AddressDetector.MEDIUM_NEGATIVE_TERMS)
        weak_hits = AddressDetector._term_hits(candidate, AddressDetector.WEAK_NEGATIVE_TERMS)
        if medium_hits >= 3 or (medium_hits >= 2 and weak_hits >= 3):
            return ""

        # RELAXED: For Aadhaar cards, don't require positive keywords if line is long enough and has digits
        has_address_keyword = AddressDetector._term_hits(candidate, AddressDetector.POSITIVE_ADDRESS_KEYWORDS) > 0
        is_long_with_digits = len(candidate) > 20 and re.search(r'\d', candidate)
        
        if not has_address_keyword and not is_long_with_digits:
            return ""

        if len(candidate) < 10 or len(candidate.split()) < 2:  # Relaxed: min 10 chars, 2 words instead of 12/3
            return ""

        return candidate
    
    def detect_and_extract(self, image_path: str) -> Optional[Dict]:
        """
        Detect address region in image and extract text via OCR.
        For Aadhaar cards, processes full image for better accuracy (YOLO is optional).
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            h, w = img.shape[:2]
            crop = img  # Default to full image for Aadhaar cards
            yolo_conf = 0.0
            used_yolo = False
            
            # Attempt YOLO detection (optional - for invoice/document localization)
            if self.model:
                try:
                    results = self.model.predict(source=image_path, conf=0.20, verbose=False)
                    if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                        used_yolo = True
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
                    print(f"Note: YOLO inference attempted but not required: {model_exc}")
                    # Continue with full image - YOLO is optional for Aadhaar

            if self.reader is None:
                print("Warning: OCR reader unavailable, skipping address extraction")
                return None

            # OCR extraction on detected crop, or full image for Aadhaar
            ocr_result = self.reader.readtext(crop, detail=1)
            
            # Always try full image if crop didn't yield results or if YOLO wasn't used
            if (not ocr_result or len(ocr_result) < 3) and used_yolo and crop is not img:
                print("Cropped region had limited results, trying full image...")
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
                print(f"No address extracted from {len(lines)} OCR lines")
                return None

            ocr_conf = float(np.mean(confs)) if confs else 0.0
            # Use only OCR confidence for Aadhaar (no YOLO weighting)
            confidence = ocr_conf
            confidence = float(max(0.0, min(1.0, confidence)))

            return {
                "image_name": Path(image_path).name,
                "address": clean_address,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
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
