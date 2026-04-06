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
            
            # Has postal code OR pincode
            # Avoid accepting a bare 6-digit line by itself (common false positive).
            has_pin = bool(re.search(r'\b\d{6}\b', line))
            is_bare_pin = bool(re.fullmatch(r'\d{6}', line.strip()))
            if (has_pin or 'pin' in line.lower() or 'postal' in line.lower()) and not is_bare_pin:
                # Keep it, but score it later (don't auto-pick numeric junk)
                postal_lines.append((idx, line))
        
        if postal_lines:
            scored = [(line, _candidate_score(line)) for _, line in postal_lines]
            result, s = max(scored, key=lambda x: x[1])
            if s >= 6.0:
                cleaned_candidate = _strip_disclaimer_phrases(result)
                print(f"  ✅ TIER 2 (Postal code): {cleaned_candidate[:80]}  [score={s:.2f}]")
                return cleaned_candidate if len(cleaned_candidate) >= 5 else ""
        
        # TIER 3: Lines with ONE address keyword + validation
        single_keyword_lines = []
        for idx, raw_line in enumerate(lines):
            line = AddressDetector._clean_text(raw_line)
            if len(line) < 8:
                continue
            
            if any(pattern.search(line) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
                continue
            if any(pattern.search(line) for pattern in AddressDetector.DOB_PATTERNS):
                continue
            
            # Has at least ONE strong address keyword AND contains numbers/punctuation (address signs)
            keyword_hit = AddressDetector._term_hits(line, AddressDetector.POSITIVE_ADDRESS_KEYWORDS) >= 1
            has_address_structure = bool(re.search(r'[,/#\-\d]', line))
            
            if keyword_hit and has_address_structure:
                print(f"  ✅ TIER 3 (Single keyword + structure): {line[:80]}")
                single_keyword_lines.append((idx, line))
        
        if single_keyword_lines:
            result = max(single_keyword_lines, key=lambda x: len(x[1]))[1]
            return AddressDetector._clean_text(result) if len(result) >= 5 else ""
        
        # TIER 4: Long lines with mixed alphanumeric + commas/slashes (address-like structure)
        # This is a fallback - accept anything that LOOKS like an address
        structure_lines = []
        for idx, raw_line in enumerate(lines):
            line = AddressDetector._clean_text(raw_line)
            if len(line) < 12:
                continue
            
            # ONLY reject absolute ID markers
            if any(pattern.search(line) for pattern in AddressDetector.DISALLOWED_ID_PATTERNS):
                continue
            if _is_bad_address_line(line):
                continue
            
            # Check for address structure: numbers + text + delimiters
            has_numbers = bool(re.search(r'\d', line))
            has_text = bool(re.search(r'[a-z]', line.lower()))
            has_delimiters = bool(re.search(r'[,/#\-]', line))
            kw_score = _weighted_keyword_score(line)
            has_pin = bool(re.search(r"\b\d{6}\b", line))
            
            # Require either keyword evidence or a PIN mention; otherwise Tier 4 is too noisy.
            if has_numbers and has_text and has_delimiters and len(line) >= 20 and (kw_score >= 1.5 or has_pin):
                print(f"  ✅ TIER 4 (Address structure): {line[:80]}")
                structure_lines.append((idx, line, kw_score, has_pin))
        
        if structure_lines:
            # Prefer higher keyword score, then PIN presence, then length
            result = max(structure_lines, key=lambda x: (x[2], 1 if x[3] else 0, len(x[1])))[1]
            return AddressDetector._clean_text(result) if len(result) >= 10 else ""
        
        print(f"  ❌ No address candidates found in any tier")
        return ""
    
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
                print(f"❌ Cannot read image: {image_path}")
                return None

            print(f"🖼️  Running OCR on {Path(image_path).name}...")
            # Run OCR on full image (fast!)
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
                print(f"❌ No address extracted from {len(lines)} OCR lines")
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
