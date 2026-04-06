import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_warp(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp image to a flat rectangle using 4 corner points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
    return warped


def detect_paper_quad(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.12,
    canny1: int = 50,
    canny2: int = 150,
) -> tuple[np.ndarray | None, dict]:
    """
    Detect a rectangular document in an image.
    Returns (quad_points or None, debug_dict).
    """
    debug = {}
    h, w = image_bgr.shape[:2]
    frame_area = float(h * w)
    min_area = min_area_ratio * frame_area

    scale = 0.5
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 11, 75, 75)
    gray_blur = cv2.GaussianBlur(gray_blur, (7, 7), 0)

    edges = cv2.Canny(gray_blur, canny1, canny2)
    kernel_close = np.ones((9, 9), np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    debug["edges"] = cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)
    debug["gray"] = cv2.resize(gray, (w, h))

    # Strategy 1: Canny edges
    quad = _find_quad_from_edges(edges, min_area * scale * scale, scale, debug)
    if quad is not None:
        debug["strategy"] = "downscale+bilateral"
        return quad, debug

    # Strategy 2: Adaptive threshold
    gray2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5,
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    quad = _find_quad_from_edges(thresh, min_area * scale * scale, scale, debug)
    if quad is not None:
        debug["strategy"] = "adaptive_threshold"
        return quad, debug

    # Strategy 3: Convex hull fallback
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:5]:
        area = cv2.contourArea(c)
        if area < min_area * scale * scale:
            continue
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        quad = (box / scale).astype(np.float32)
        bw, bh = rect[1]
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 5.0:
            continue
        debug["strategy"] = "convex_hull_minAreaRect"
        return quad, debug

    return None, debug


def _find_quad_from_edges(edges, min_area, scale, debug):
    """Find a 4-point quadrilateral from an edge image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:20]:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)

        approx = None
        for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            candidate = cv2.approxPolyDP(c, eps * peri, True)
            if len(candidate) == 4:
                approx = candidate
                break

        if approx is None:
            continue

        quad = (approx.reshape(4, 2) / scale).astype(np.float32)

        rect = cv2.boundingRect(approx)
        _, _, bw, bh = rect
        aspect = bw / float(bh + 1e-6)
        if aspect < 0.15 or aspect > 6.0:
            continue

        return quad

    return None
