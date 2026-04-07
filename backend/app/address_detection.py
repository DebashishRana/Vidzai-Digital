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
        "bypass",
        "road",
        "ring road",
    }

    # Higher-priority keyword buckets specifically for Aadhaar address blocks.
    # Used for weighted scoring (not hard-rejection).
    AADHAAR_ADDRESS_KEYWORDS = {
        "residential": {
            "house", "h.no", "hno", "house no",
            "plot", "plot no",
            "door no", "d.no",
            "flat", "floor",
            "apartment", "building",
            "block", "sector",
        },
        "locality": {
            "nagar", "colony", "layout",
            "mohalla", "moh",
            "area", "locality",
            "phase", "extension", "ext",
        },
        "rural": {
            "village", "vill", "gaon",
            "panchayat",
            "tehsil", "taluk", "mandal",
            "district", "dist",
            "khasra", "khasra no",
            "khata",
            "ward", "ward no",
        },
        "postal": {
            "post", "post office", "po",
            "sub post office", "spo",
            "head post office", "hpo",
            "pin", "pincode", "pin code",
        },
        "road": {
            "road", "rd",
            "street", "st",
            "lane",
            "marg", "path",
            "main", "cross",
            "highway", "nh", "sh",
            "bypass", "ring road",
        },
        "landmark": {
            "near", "nearby",
            "behind",
            "opposite", "opp",
            "beside",
            "in front of",
            "landmark",
        },
        "building": {
            "tower", "complex",
            "residency", "residence",
            "villa", "villas",
            "heights",
            "plaza",
            "mall",
            "market",
            "chowk",
        }
    }

    _AADHAAR_KW_WEIGHTS = {
        "residential": 4.0,
        "postal": 3.5,
        "road": 3.0,
        "locality": 2.5,
        "rural": 2.5,
        "landmark": 2.0,
        "building": 2.0,
    }

    _LINE_BLACKLIST_PHRASES = {
        "aadhaar app",
        "secure qr",
        "qr code",
        "offline xml",
        "xml",
        "enrolment",
        "enrollment",
        "enrolment no",
        "enrollment no",
        "unique identification authority",
        "uidai",
        "government of india",
        "information",
        "authentication",
        "digitally signed",
        "download date",
        "issue date",
        "help@uidai",
        "www.uidai.gov.in",
        # Common Aadhaar disclaimer text fragments (often leaks into address OCR)
        "electronically generated",
        "proof of identity",
        "not of citizenship",
        "to establish identity",
        "authenticate online",
        "enrolment",
        "enrollment",
        "enrolment no",
        "enrollment no",
        "virtual id",
        "vid",
    }

    # Regexes to strip from merged OCR address candidates (boilerplate + OCR artifacts).
    _NOISE_REGEXES = [
        # Aadhaar boilerplate / disclaimers
        r"\bthis\s+is\s+electronically\s+generated\s+letter\b",
        r"\baadhaar\s+is\s+a\s+proof\s+of\s+identity\b",
        r"\bnot\s+of\s+citizenship\b",
        r"\bto\s+establish\s+identity\b",
        r"\bauthenticate\s+online\b",
        r"\bvirtual\s+id\b",
        r"\bvid\b",
        r"\benrol(?:ment|lment)\s*(?:no|number)?\b",
        r"\benrol(?:ment|lment)\s*(?:no|number)?\s*[:\-]?\s*[\w/.\-]+\b",
        r"\benroll(?:ment|lment)\s*(?:no|number)?\b",
        r"\benroll(?:ment|lment)\s*(?:no|number)?\s*[:\-]?\s*[\w/.\-]+\b",
        r"\bunique\s+identification\s+authority\s+of\s+india\b",
        r"\buidai\b",
        r"\bgovernment\s+of\s+india\b",
        r"\binformation\b",
        # Common OCR artifacts / stray tokens
        r"\bf\s*-\b",
        r"\be1r\b",
        r"\b7uat\b",
        r"\b7uat\}\b",
        r"\b[a-z]\d[a-z]\b",  # e.g. e1r
        r"\b\d[a-z]{2,}\b",   # digit + letters chunks
    ]

    _INDIA_STATE_CITY_HINTS = {
        # States/UTs (common in addresses)
        "maharashtra", "gujarat", "karnataka", "tamil nadu", "tamilnadu", "kerala", "telangana",
        "andhra pradesh", "uttar pradesh", "madhya pradesh", "rajasthan", "punjab", "haryana",
        "west bengal", "bihar", "odisha", "assam", "jharkhand", "chhattisgarh", "goa",
        "delhi", "new delhi",
        # Cities/areas (lightweight hints; not exhaustive)
        "pune", "mumbai", "nagpur", "nashik", "thane", "bengaluru", "bangalore", "hyderabad",
        "chennai", "kolkata", "ahmedabad", "surat", "jaipur", "lucknow",
        "wagholi",
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
        # YOLO is NOT used anymore - skipping
        self.model = None
        print("YOLO skipped (not needed for Aadhaar address extraction)")

        print("Initializing EasyOCR...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"Warning: EasyOCR init failed: {str(e)}")
            self.reader = None
        
        print("✅ Address detector initialized (OCR-based, YOLO-free)")

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
        """
        Ultra-robust address extraction using aggressive keyword matching.
        Strategy: ACCEPT aggressively if address-like, REJECT only if clearly not.
        """
        if not lines:
            _dbg("H5", "backend/app/address_detection.py:_extract_address_from_lines", "No OCR lines to score", {"lines": 0})
            return ""
        
        print(f"🔍 Processing {len(lines)} OCR lines for address extraction...")

        # TIER 0: Aadhaar "To" block heuristic (most reliable for Aadhaar letters/cards)
        # If we see a "To" marker, scan forward until we hit a PIN line and join the
        # non-noise lines along the way.
        cleaned = [AddressDetector._clean_text(l) for l in lines]
        to_idxs = []
        for i, l in enumerate(cleaned):
            low = (l or "").strip().lower()
            if low == "to" or low.startswith("to "):
                to_idxs.append(i)

        def _looks_like_pin_line(s: str) -> bool:
            return bool(re.search(r"\b\d{6}\b", s or ""))

        def _is_bad_address_line(s: str) -> bool:
            low = (s or "").lower()
            if not s or len(s) < 2:
                return True
            if any(p.search(s) for p in AddressDetector.DISALLOWED_ID_PATTERNS):
                return True
            if any(p.search(s) for p in AddressDetector.DOB_PATTERNS):
                return True
            # For Aadhaar, treat common header/footer phrases as noise lines,
            # but do not globally reject the whole candidate later.
            for phrase in AddressDetector._LINE_BLACKLIST_PHRASES:
                if phrase in low:
                    return True
            return False

        def _weighted_keyword_score(text: str) -> float:
            t = (text or "").lower()
            score = 0.0
            # Weighted Aadhaar keyword buckets
            for bucket, kws in AddressDetector.AADHAAR_ADDRESS_KEYWORDS.items():
                hits = sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", t))
                if hits:
                    score += hits * float(AddressDetector._AADHAAR_KW_WEIGHTS.get(bucket, 1.0))
            # Generic address keyword presence (small boost)
            score += 0.5 * float(AddressDetector._term_hits(text, AddressDetector.POSITIVE_ADDRESS_KEYWORDS))
            # Boost common India state/city hints (helps when OCR misses keywords)
            score += 1.25 * sum(1 for h in AddressDetector._INDIA_STATE_CITY_HINTS if re.search(rf"\b{re.escape(h)}\b", t))
            return score

        def _alpha_ratio(text: str) -> float:
            s = text or ""
            if not s:
                return 0.0
            alpha = sum(1 for c in s if c.isalpha())
            digits = sum(1 for c in s if c.isdigit())
            denom = max(1, alpha + digits)
            return alpha / denom

        def _candidate_score(text: str) -> float:
            t = (text or "").strip()
            low = t.lower()
            if not t:
                return -1e9
            # Strongly penalize footer/header noise if it slipped in
            if any(p in low for p in AddressDetector._LINE_BLACKLIST_PHRASES):
                return -50.0
            # Prefer multi-line / longer but not pure gibberish
            kw = _weighted_keyword_score(t)
            has_pin = 3.5 if re.search(r"\b\d{6}\b", t) else 0.0
            ar = _alpha_ratio(t)
            digit_heavy_penalty = 6.0 * max(0.0, 0.55 - ar)  # penalize low alpha ratio
            length_bonus = min(8.0, len(t) / 30.0)
            return (kw * 2.0) + has_pin + length_bonus - digit_heavy_penalty

        def _strip_disclaimer_phrases(text: str) -> str:
            """
            Remove common Aadhaar disclaimer fragments that sometimes get OCR-merged
            into the address lines (without dropping the entire address).
            """
            t = AddressDetector._clean_text(text)
            for p in AddressDetector._NOISE_REGEXES:
                t = re.sub(p, " ", t, flags=re.IGNORECASE)

            # Collapse spaces and trim again.
            t = re.sub(r"\s+", " ", t).strip()
            return t

        for to_i in to_idxs[:2]:
            window = []
            pin_found = False
            # OCR order can be noisy; give a wider scan window.
            for j in range(to_i + 1, min(len(cleaned), to_i + 45)):
                s = cleaned[j]
                if _is_bad_address_line(s):
                    continue
                window.append(s)
                if _looks_like_pin_line(s):
                    pin_found = True
                    break

            # If PIN is on its own line (just 6 digits), also include prior 1-2 lines
            if pin_found and window:
                candidate = " ".join(window).strip()
                # Require non-trivial address-likeness beyond just PIN
                has_delims = bool(re.search(r"[,/#\-]", candidate))
                has_numbers = bool(re.search(r"\d", candidate))
                kw_score = _weighted_keyword_score(candidate)
                if (has_numbers and (_candidate_score(candidate) >= 6.0 or (kw_score >= 2.0 and has_delims))) and len(candidate) >= 12:
                    cleaned_candidate = _strip_disclaimer_phrases(candidate)
                    print(f"  ✅ TIER 0 (To-block): {cleaned_candidate[:80]}  [score={_candidate_score(candidate):.2f}]")
                    return cleaned_candidate
        
        # TIER 1: Look for lines with STRONG address keywords (highest confidence)
        strong_address_lines = []
        for idx, raw_line in enumerate(lines):
            line = AddressDetector._clean_text(raw_line)
            if len(line) < 3:
                continue
            
            # REJECT only: Clear ID markers
            if any(pattern.search(line) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
                continue
            
            # REJECT only: 12-digit Aadhaar, DOB patterns
            if any(pattern.search(line) for pattern in AddressDetector.DOB_PATTERNS):
                continue
            
            # STRONG ACCEPT: Contains MULTIPLE address keywords
            keyword_hits = AddressDetector._term_hits(line, AddressDetector.POSITIVE_ADDRESS_KEYWORDS)
            if keyword_hits >= 2:
                print(f"  ✅ TIER 1 (Multiple keywords): {line[:80]}")
                strong_address_lines.append((idx, line, keyword_hits))
        
        if strong_address_lines:
            result = max(strong_address_lines, key=lambda x: (x[2], len(x[1])))[1]
            return AddressDetector._clean_text(result) if len(result) >= 5 else ""
        
        # TIER 2: PIN-anchored window candidates (most reliable when "To" scanning fails)
        # Build candidates around any line that contains a 6-digit PIN by joining nearby non-noise lines.
        pin_candidates: list[str] = []
        pin_idxs = [i for i, l in enumerate(cleaned) if re.search(r"\b\d{6}\b", l or "")]
        for pin_i in pin_idxs[:6]:
            chunk: list[str] = []
            for j in range(max(0, pin_i - 6), min(len(cleaned), pin_i + 2)):
                s = cleaned[j]
                if _is_bad_address_line(s):
                    continue
                chunk.append(s)
            if chunk:
                pin_candidates.append(AddressDetector._clean_text(" ".join(chunk)))

        if pin_candidates:
            scored = [(c, _candidate_score(c)) for c in pin_candidates]
            best_c, best_s = max(scored, key=lambda x: x[1])
            # Only accept if it looks address-like (filters out numeric junk like "314641 ...")
            if best_s >= 6.0 and len(best_c.split()) >= 3:
                cleaned_candidate = _strip_disclaimer_phrases(best_c)
                print(f"  ✅ TIER 2 (PIN-window): {cleaned_candidate[:80]}  [score={best_s:.2f}]")
                return cleaned_candidate

        # Legacy TIER 2: Lines with postal code + other address markers (kept as fallback)
        postal_lines = []
        for idx, raw_line in enumerate(lines):
            line = AddressDetector._clean_text(raw_line)
            if len(line) < 5:
                continue
            
            if any(pattern.search(line) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
                continue
            if any(pattern.search(line) for pattern in AddressDetector.DOB_PATTERNS):
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
        SIMPLE & FAST: Extract address from any document using OCR + text logic.
        Works on Aadhaar, PAN, invoices - NO YOLO model needed anymore.
        """
        try:
            if self.reader is None:
                print("❌ Warning: OCR reader unavailable")
                return None

            img = cv2.imread(image_path)
            if img is None:
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
                return None

            # OCR extraction on detected crop, with full-image fallback if needed
            ocr_result = self.reader.readtext(crop, detail=1)
            if not ocr_result and crop is not img:
                ocr_result = self.reader.readtext(img, detail=1)

            lines = []
            confs = []
            for item in ocr_result:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    if text:
                        lines.append(text)
                    try:
                        confs.append(float(item[2]))
                    except Exception:
                        pass

            print(f"📄 OCR extracted {len(lines)} text lines:")
            for i, line in enumerate(lines[:20]):  # Show first 20 lines
                print(f"   [{i}] {line[:100]}")
            if len(lines) > 20:
                print(f"   ... and {len(lines) - 20} more lines")

            # Extract address using logic (no YOLO required)
            print("\n🔎 Starting address extraction logic...")
            clean_address = self._extract_address_from_lines(lines)
            
            if not clean_address:
                return None

            print(f"\n✅ Address Found: {clean_address}")
            # Confidence from OCR only
            confidence = float(np.mean(confs)) if confs else 0.75
            confidence = float(max(0.0, min(1.0, confidence)))

            return {
                "image_name": Path(image_path).name,
                "address": clean_address,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
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
