import os
import cv2
import csv
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re

# NEW: optional NEF support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("[warning] rawpy not installed – NEF files will not be readable.")


try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# =========================
# TOOL VERSION
# =========================
TOOL_VERSION = "0.6.1"  # Sensitivity tuning: percentile 95, min-ceiling threshold, no morph cleanup

# =========================
# GLOBAL TUNING CONSTANTS
# =========================

# User-guided ROI: pixels to shrink inward from each edge after corner clicks
ROI_CORNER_SHRINK_PX = 5

# Legacy auto-circle constants (kept for reference; auto-detect no longer used in main flow)
AUTO_INNER_SCALE = 0.96
INNER_SHRINK = 5

# Only local-contrast pixels above this percentile (inside ROI) are dust
DUST_PERCENTILE = 92.0      # higher -> fewer pixels marked as dust

# Baseline (untreated) comparison tuning for dust detection
# Lower thresholds here make the detector more sensitive to dark specks
# relative to the baseline patches you selected.
# Slightly more aggressive values (K and absolute delta) to better catch
# mid-tone "shadowy" dust regions while still rejecting most noise.
BASELINE_SIGMA_K = 0.3         # smaller K => more sensitive to darker-than-baseline pixels
BASELINE_MIN_ABS_DELTA = 0.7   # allow moderately darker specks/shadows to count as dust
BASELINE_LOCAL_PERCENTILE = 85.0  # slightly lower so more local-contrast specks qualify

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef")

def load_image_any(img_path: str):
    """
    Load an image from disk.
    - For standard formats, uses cv2.imread (BGR).
    - For .NEF raw files, uses rawpy to decode to RGB, then converts to BGR.
    """
    ext = os.path.splitext(img_path)[1].lower()

    if ext == ".nef":
        if not HAS_RAWPY:
            raise RuntimeError(
                "Tried to open a NEF file but rawpy is not installed.\n"
                "Install it with: pip install rawpy"
            )
        # NEF: decode RAW to 16-bit RGB, then downscale to 8-bit and convert to BGR for OpenCV
        with rawpy.imread(img_path) as raw:
            flip = raw.sizes.flip          # read EXIF orientation before closing
            rgb16 = raw.postprocess(
                use_camera_wb=True,        # honor camera white balance
                no_auto_bright=True,       # avoid aggressive tone curve that boosts noise
                output_bps=16,             # keep 16-bit internal, then downscale ourselves
                gamma=(2.222, 4.5),        # sRGB gamma — matches how the eye expects images to look
                user_flip=0,               # we apply rotation ourselves below
            )
        # Convert 16-bit RGB to 8-bit RGB, then to BGR for OpenCV
        rgb8 = (rgb16 / 256).astype(np.uint8)
        bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        # Apply EXIF/camera orientation (rawpy flip codes match LibRaw convention)
        if flip == 3:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        elif flip == 5:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif flip == 6:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        return bgr
    else:
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        return img

# =========================
# AUTO CIRCLE DETECTION
# =========================

def find_ring_mask_auto(image_bgr, inner_fraction: float = 0.44, inner_shrink: int = 10, debug: bool = False):
    """
    Auto-detect circular ROI inside the colored ring.

    Strategy:
      1) Try to find the colored ring (yellow / orange / green / pink) in HSV.
      2) Fit a circle to that ring and set ROI radius = ring_radius * inner_fraction - inner_shrink.
      3) If color-based detection fails (e.g. grayscale images), fall back to HoughCircles
         on grayscale and use the smallest plausible circle as ROI.

    Returns:
      mask_roi (uint8 0/255), circle = (cx, cy, r_roi)
    """

    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    img_cx, img_cy = w / 2.0, h / 2.0

    # -----------------------
    # 1) COLOR-BASED RING FIND
    # -----------------------
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # HSV ranges for known ring colors (rough; can be tuned)
    color_ranges = [
        # yellow
        (np.array([15, 80, 80], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)),
        # orange
        (np.array([5, 80, 80], dtype=np.uint8),  np.array([20, 255, 255], dtype=np.uint8)),
        # green
        (np.array([40, 40, 40], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)),
        # pink / magenta (wraps around 180, so use two ranges)
        (np.array([160, 60, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
        (np.array([0, 60, 60], dtype=np.uint8),   np.array([10, 255, 255], dtype=np.uint8)),
    ]

    best_ring = None
    best_score = -1e9

    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)

        # clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < (min_dim * min_dim * 0.002):  # ignore tiny blobs
                continue

            (cx, cy), r = cv2.minEnclosingCircle(c)
            cx, cy, r = float(cx), float(cy), float(r)

            # center distance from image center (normalized)
            dc = np.hypot(cx - img_cx, cy - img_cy) / min_dim
            if dc > 0.35:  # too far from center → probably tape/background
                continue

            # radius constraints: ring should be fairly big but not huge
            if not (min_dim * 0.08 < r < min_dim * 0.35):
                continue

            # score: prefer large area and centered
            score = area - (dc * 5000.0)
            if score > best_score:
                best_score = score
                best_ring = (cx, cy, r)

    # -----------------------
    # 2) FALLBACK: HoughCircles if no ring found
    # -----------------------
    if best_ring is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 4,
            param1=120,
            param2=35,
            minRadius=int(min_dim * 0.05),
            maxRadius=int(min_dim * 0.40),
        )

        if circles is None:
            raise RuntimeError("Auto mode: no suitable circle found (color + Hough failed).")

        circles = np.round(circles[0]).astype(int)
        # smallest plausible circle as ROI fallback
        circles = sorted(circles, key=lambda c: c[2])
        cx, cy, r = circles[0]
        best_ring = (float(cx), float(cy), float(r))

    # -----------------------
    # 3) Build ROI mask inside the ring
    # -----------------------
    cx, cy, r_ring = best_ring

    # ROI radius: fraction of ring radius minus optional shrink
    r_roi = int(r_ring * inner_fraction) - inner_shrink
    r_roi = max(5, r_roi)

    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_roi, (int(cx), int(cy)), r_roi, 255, -1)

    if debug:
        # visual check: show ring circle and ROI circle
        dbg = cv2.cvtColor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.circle(dbg, (int(cx), int(cy)), int(r_ring), (0, 255, 0), 2)   # green: ring
        cv2.circle(dbg, (int(cx), int(cy)), int(r_roi), (255, 0, 0), 2)    # blue: ROI
        cv2.imshow("Auto ROI (ring + fallback)", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_roi, (int(cx), int(cy), int(r_roi))


# =========================
# DISC DETECTION HELPER
# =========================

def _find_disc_center(image_bgr):
    """
    Locate the coloured filter disc (orange / yellow / green / pink) in image_bgr.

    Uses HSV colour segmentation across multiple colour ranges, then applies a
    circularity filter  (circularity = 4*pi*area / perimeter^2 > 0.45)  to
    reject elongated tape strips that share the disc's hue but are not circular.

    Returns: (cx, cy, r)  -- centre x, centre y, and radius in pixels (floats)
    Raises:  RuntimeError if no sufficiently circular blob is found.
    """
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    img_cx, img_cy = w / 2.0, h / 2.0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_ranges = [
        (np.array([15, 80, 80],  dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)),  # yellow/orange
        (np.array([5,  80, 80],  dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)),  # orange
        (np.array([40, 40, 40],  dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)),  # green
        (np.array([160, 60, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)), # pink/red
        (np.array([0,  60, 60],  dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),  # pink/red wrap
    ]

    best = None
    best_score = -1e9
    kernel = np.ones((5, 5), np.uint8)

    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < (min_dim * min_dim * 0.002):
                continue

            # --- circularity filter: rejects tape strips and other elongated blobs ---
            perimeter = cv2.arcLength(c, True)
            if perimeter < 1:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < 0.45:
                continue  # not round enough -- skip

            (cx_c, cy_c), r_c = cv2.minEnclosingCircle(c)
            cx_c, cy_c, r_c = float(cx_c), float(cy_c), float(r_c)

            if not (min_dim * 0.04 < r_c < min_dim * 0.55):
                continue

            # Soft penalty for being far from image centre
            dc = np.hypot(cx_c - img_cx, cy_c - img_cy) / min_dim
            score = area - (dc * 3000.0)

            if score > best_score:
                best_score = score
                best = (cx_c, cy_c, r_c)

    if best is None:
        raise RuntimeError(
            "_find_disc_center: could not detect a circular coloured disc. "
            "Check that the sample is visible and not obscured by tape or other objects."
        )

    return best  # (cx, cy, r)


# =========================
# AUTO-DETECT SQUARE ROI FROM GRID
# =========================

def find_roi_from_grid(image_bgr, shrink=ROI_CORNER_SHRINK_PX):
    """
    Detect the ROI square from the printed black grid lines on the target card.

    Strategy: morphological line detection + run-width filtering.

    1. Threshold at 50 to isolate near-black ink (sRGB black ink < 30,
       dark metallic surfaces typically > 60 after gamma).
    2. Mask out the outer 5 % border so metallic frame edges are ignored.
    3. MORPH_OPEN with a long horizontal kernel --> keeps only long horizontal
       strokes; short blobs and vertical features are erased.
       MORPH_OPEN with a long vertical kernel   --> keeps only long vertical
       strokes; short blobs and horizontal features are erased.
    4. Project each result onto its axis (count dark pixels per row/column).
    5. Group consecutive dark rows/columns into "runs":
         - NARROW runs (a few px wide) = printed grid lines   --> keep
         - WIDE runs  (many px tall)   = dark background bands --> discard
    6. The outermost surviving narrow runs define the grid outer boundary = ROI.

    Returns: (mask_roi uint8 0/255, (cx, cy, half_side))
    Raises:  RuntimeError if at least 2 H-lines + 2 V-lines cannot be found.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Step 1: threshold ---
    # Use 70 (generous) so Sharpie ink is captured even where the disc sits on
    # top of it.  Fine metallic lines are kept too, but the run-width filter
    # below removes them.
    _, dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # --- Step 2: ignore the outer border strip (prevents metallic rim detections) ---
    bdr = max(10, int(min(h, w) * 0.05))
    dark[:bdr, :] = 0
    dark[-bdr:, :] = 0
    dark[:, :bdr] = 0
    dark[:, -bdr:] = 0

    # --- Step 3: morphological line extraction ---
    # Sharpie lines can be partially hidden by the disc, so use a shorter
    # minimum length (1/8 of dimension) to catch even broken segments.
    h_len = max(20, w // 8)
    v_len = max(20, h // 8)
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert  = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))

    # --- Step 4: row/column projections ---
    h_count = np.count_nonzero(horiz, axis=1)   # dark pixels per row
    v_count = np.count_nonzero(vert,  axis=0)   # dark pixels per column

    h_rows = np.where(h_count > w * 0.05)[0]    # 5%: catches partially-visible lines
    v_cols = np.where(v_count > h * 0.05)[0]

    # --- Step 5: run-width band-pass filter ---
    # The Sharpie marker makes THICK lines.  Fine metallic lines are thin.
    # Dark metallic background bands are very wide.
    #
    #   run width < MIN_PX  --> fine metallic line  --> discard
    #   MIN_PX <= run <= MAX_PX --> Sharpie grid line --> KEEP
    #   run width > MAX_PX  --> dark background band --> discard
    #
    # After the morphological OPEN each printed Sharpie line shows as a run
    # of consecutive dark rows/columns a few to ~60 px wide.
    MIN_LINE_PX = 4    # thinner than this = fine metallic line
    MAX_LINE_PX = 80   # wider than this = dark background band

    def _gridline_centers(positions):
        if len(positions) == 0:
            return []
        lines = []
        run = [int(positions[0])]
        for p in positions[1:]:
            if int(p) - run[-1] <= 3:
                run.append(int(p))
            else:
                rw = run[-1] - run[0] + 1
                if MIN_LINE_PX <= rw <= MAX_LINE_PX:
                    lines.append(int(round(np.mean(run))))
                run = [int(p)]
        rw = run[-1] - run[0] + 1
        if MIN_LINE_PX <= rw <= MAX_LINE_PX:
            lines.append(int(round(np.mean(run))))
        return lines

    h_lines = _gridline_centers(h_rows)
    v_lines = _gridline_centers(v_cols)

    if len(h_lines) < 2:
        raise RuntimeError(
            "find_roi_from_grid: fewer than 2 horizontal grid lines found. "
            "Make sure the grid card is visible and the image is in focus."
        )
    if len(v_lines) < 2:
        raise RuntimeError(
            "find_roi_from_grid: fewer than 2 vertical grid lines found. "
            "Make sure the grid card is visible and the image is in focus."
        )

    # --- Step 6: outer boundary of surviving narrow runs = grid square ---
    y_top   = min(h_lines)
    y_bot   = max(h_lines)
    x_left  = min(v_lines)
    x_right = max(v_lines)

    # Sanity: grid must be a meaningful portion of the image
    if (y_bot - y_top) < h * 0.10 or (x_right - x_left) < w * 0.10:
        raise RuntimeError(
            f"find_roi_from_grid: detected grid region too small "
            f"({x_right - x_left}x{y_bot - y_top} px in {w}x{h} image). "
            "Check that the full grid is in frame."
        )

    # Shrink inward past the line pixels themselves
    y_top   = min(h - 1, y_top   + shrink)
    y_bot   = max(0,     y_bot   - shrink)
    x_left  = min(w - 1, x_left  + shrink)
    x_right = max(0,     x_right - shrink)

    cx_roi    = (x_left + x_right) // 2
    cy_roi    = (y_top  + y_bot)   // 2
    half_side = min((x_right - x_left) // 2, (y_bot - y_top) // 2)

    mask_roi = np.zeros((h, w), dtype=np.uint8)
    mask_roi[y_top : y_bot + 1, x_left : x_right + 1] = 255

    return mask_roi, (cx_roi, cy_roi, half_side)


def straighten_and_crop_to_card(image_bgr, padding_px=40):
    """
    Detect the white target card, rotate the image so grid lines are
    horizontal/vertical at ANY tilt angle, then crop tightly to the card.

    Key improvement over the previous version: card selection is ANCHORED to
    the disc location.  After finding the disc with _find_disc_center(), we
    pick the white contour that CONTAINS the disc centre (pointPolygonTest)
    rather than blindly choosing the largest white blob.  This prevents shiny
    metallic surfaces from being mistaken for the paper target card.

    Steps:
      1. Find disc centre with _find_disc_center().
      2. HSV-segment white/light-grey areas.
      3. Select the white contour that contains the disc centre.
         Fall back to largest contour if none contains the disc.
      4. Rotate by minAreaRect angle -- works at any tilt.
      5. Crop to card bounding box in the rotated image (disc-anchored again).

    Returns: (processed_image, correction_angle_deg)
    Falls back to (original_image, 0.0) if the card cannot be detected.
    """
    h, w = image_bgr.shape[:2]

    # --- Step 1: locate disc ---
    try:
        disc_cx, disc_cy, _disc_r = _find_disc_center(image_bgr)
    except RuntimeError as exc:
        print(f"[straighten] {exc} -- skipping rotation/crop.")
        return image_bgr, 0.0

    # --- Step 2: find white/light-grey regions ---
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv,
                             np.array([0,   0, 150], dtype=np.uint8),
                             np.array([180, 80, 255], dtype=np.uint8))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kern)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kern)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[straighten] Could not detect white target card -- skipping rotation/crop.")
        return image_bgr, 0.0

    # --- Step 3: pick white contour that contains the disc centre ---
    disc_pt = (float(disc_cx), float(disc_cy))
    card_contour = None
    large_contours = [c for c in contours if cv2.contourArea(c) >= h * w * 0.04]

    for c in large_contours:
        if cv2.pointPolygonTest(c, disc_pt, False) >= 0:
            card_contour = c
            break

    if card_contour is None:
        if not large_contours:
            print("[straighten] No large white region found -- skipping rotation/crop.")
            return image_bgr, 0.0
        card_contour = max(large_contours, key=cv2.contourArea)
        print("[straighten] Disc centre not inside any white region; using largest white blob.")

    # --- Step 4: compute rotation angle and rotate ---
    rect = cv2.minAreaRect(card_contour)
    rect_angle = rect[2]   # in [-90, 0)

    # Pick the candidate rotation closest to 0 (smallest absolute correction)
    cand1 = rect_angle
    cand2 = rect_angle + 90.0
    correction = cand1 if abs(cand1) <= abs(cand2) else cand2

    cos_a = abs(np.cos(np.radians(correction)))
    sin_a = abs(np.sin(np.radians(correction)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, correction, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    rotated = cv2.warpAffine(image_bgr, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(128, 128, 128))

    # --- Step 5: crop to card in rotated image (disc-anchored) ---
    rh, rw = rotated.shape[:2]
    hsv2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    white_mask2 = cv2.inRange(hsv2,
                              np.array([0,   0, 150], dtype=np.uint8),
                              np.array([180, 80, 255], dtype=np.uint8))
    white_mask2 = cv2.morphologyEx(white_mask2, cv2.MORPH_CLOSE, kern)
    white_mask2 = cv2.morphologyEx(white_mask2, cv2.MORPH_OPEN, kern)
    c2, _ = cv2.findContours(white_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Transform disc centre into rotated-image coordinates
    disc_rot = M @ np.array([disc_cx, disc_cy, 1.0])
    disc_rx, disc_ry = float(disc_rot[0]), float(disc_rot[1])
    disc_pt2 = (disc_rx, disc_ry)

    crop_contour = None
    if c2:
        large2 = [c for c in c2 if cv2.contourArea(c) >= rh * rw * 0.04]
        for c in large2:
            if cv2.pointPolygonTest(c, disc_pt2, False) >= 0:
                crop_contour = c
                break
        if crop_contour is None and large2:
            crop_contour = max(large2, key=cv2.contourArea)

    if crop_contour is not None:
        x, y, bw, bh = cv2.boundingRect(crop_contour)
        x1 = max(0, x - padding_px)
        y1 = max(0, y - padding_px)
        x2 = min(rw, x + bw + padding_px)
        y2 = min(rh, y + bh + padding_px)
        rotated = rotated[y1:y2, x1:x2]

    return rotated, correction


# =========================
# WINDOW UTILITIES
# =========================

# Module-level cache so tkinter is only queried once per session.
_screen_size_cache = [None]   # [0] = (screen_w, screen_h) or None

def _center_window(win_name, win_w, win_h):
    """
    Move an OpenCV named window to the centre of the primary monitor.
    Falls back to 1920×1080 if the screen size cannot be determined.
    Must be called AFTER cv2.imshow() so the window is mapped on macOS.
    """
    if _screen_size_cache[0] is None:
        sw, sh = 1920, 1080   # safe fallback
        if TK_AVAILABLE:
            try:
                root = tk.Tk()
                root.withdraw()
                sw = root.winfo_screenwidth()
                sh = root.winfo_screenheight()
                root.destroy()
            except Exception:
                pass
        _screen_size_cache[0] = (sw, sh)
    sw, sh = _screen_size_cache[0]
    cv2.moveWindow(win_name, max(0, (sw - win_w) // 2), max(0, (sh - win_h) // 2))


# =========================
# ROTATION HELPERS
# =========================

def apply_rotation(image_bgr, angle_deg):
    """
    Rotate image_bgr by angle_deg (positive = CCW in OpenCV convention).
    Canvas is expanded so no pixel content is clipped.
    Returns the rotated image unchanged if angle is essentially zero.
    """
    if abs(angle_deg) < 0.001:
        return image_bgr
    h, w = image_bgr.shape[:2]
    cos_a = abs(np.cos(np.radians(angle_deg)))
    sin_a = abs(np.sin(np.radians(angle_deg)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image_bgr, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(128, 128, 128))


def interactive_rotation(image_bgr):
    """
    Show the image in an interactive window and let the user rotate it with
    keyboard keys until the grid lines are perfectly horizontal/vertical.
    The confirmed angle is then applied to every image in the batch.

    Keys:
        a  /  Left  arrow  --  rotate CCW  1.0 deg
        d  /  Right arrow  --  rotate  CW  1.0 deg
        z  /  ,            --  rotate CCW  0.1 deg  (fine)
        x  /  .            --  rotate  CW  0.1 deg  (fine)
        r                  --  reset to 0 deg
        Enter              --  confirm and use this angle
        q                  --  cancel (use 0 deg, no rotation)

    Returns: angle_deg (float, positive = CCW correction applied to every image)
    """
    WIN = "Straighten -- a/d to rotate  z/x for fine  r=reset  Enter=confirm  q=cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    h, w = image_bgr.shape[:2]
    MAX_W, MAX_H = 1400, 900
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    disp_w = max(1, int(w * scale))
    disp_h = max(1, int(h * scale))

    angle = 0.0

    # Pre-build the static alignment grid (never rotates -- fixed to the screen).
    # Fine lines every ~step px; brighter centre crosshair for reference.
    step = max(40, disp_h // 10)   # ~10 rows across the window height
    grid_layer = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    GRID_COLOR  = (0, 180, 180)   # dim cyan for minor grid lines
    CROSS_COLOR = (0, 255, 255)   # bright cyan for centre crosshair
    for gx in range(0, disp_w, step):
        cv2.line(grid_layer, (gx, 0), (gx, disp_h - 1), GRID_COLOR, 1)
    for gy in range(0, disp_h, step):
        cv2.line(grid_layer, (0, gy), (disp_w - 1, gy), GRID_COLOR, 1)
    # Centre crosshair (thicker + brighter)
    cv2.line(grid_layer, (disp_w // 2, 0), (disp_w // 2, disp_h - 1), CROSS_COLOR, 2)
    cv2.line(grid_layer, (0, disp_h // 2), (disp_w - 1, disp_h // 2), CROSS_COLOR, 2)

    def _render(ang):
        # Rotate the image content (the grid stays fixed on top)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        rot = cv2.warpAffine(image_bgr, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(128, 128, 128))
        disp = cv2.resize(rot, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Blend static grid over the rotated image (40% opacity)
        disp = cv2.addWeighted(disp, 1.0, grid_layer, 0.4, 0)

        # Text overlay
        lines = [
            f"Angle: {ang:+.1f} deg  |  Align card lines to the cyan grid",
            "a/Left: CCW 1deg   d/Right: CW 1deg",
            "z/,: CCW 0.1deg    x/.: CW 0.1deg",
            "r: reset   Enter: confirm   q: cancel (no rotation)",
        ]
        for i, txt in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(disp, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return disp

    cv2.imshow(WIN, _render(angle))
    cv2.resizeWindow(WIN, disp_w, disp_h)
    _center_window(WIN, disp_w, disp_h)

    while True:
        key = cv2.waitKey(0)
        k = key & 0xFFFF  # keep 16 bits for macOS extended arrow key codes

        if k in (13, 10):       # Enter / Return -- confirm
            break
        elif k == ord('q'):     # cancel -- no rotation
            angle = 0.0
            break
        elif k == ord('r'):     # reset
            angle = 0.0
        elif k in (ord('a'), 81, 63234):   # 'a' or Left arrow (81=Win, 63234=Mac)
            angle += 1.0    # CCW
        elif k in (ord('d'), 83, 63235):   # 'd' or Right arrow (83=Win, 63235=Mac)
            angle -= 1.0    # CW
        elif k in (ord('z'), ord(',')):    # fine CCW
            angle += 0.1
        elif k in (ord('x'), ord('.')):    # fine CW
            angle -= 0.1

        cv2.imshow(WIN, _render(angle))

    cv2.destroyWindow(WIN)
    return round(angle, 1)


# =========================
# USER-GUIDED SQUARE ROI
# =========================

def find_roi_user_guided(image_bgr, shrink=ROI_CORNER_SHRINK_PX):
    """
    Interactive square ROI selection.

    Shows the image in a resizable window and asks the user to click the
    4 interior corners of the 1 cm x 1 cm square target (in any order).
    A magnifier loupe assists with precise placement.

    After 4 clicks the bounding rectangle is shrunk inward by `shrink`
    pixels on every side and used as the ROI.

    Press 'r' at any time to clear clicks and start over.
    Press ENTER to confirm once all 4 corners have been clicked.

    Returns:
        mask_roi  – uint8 (h, w) array, 255 inside ROI, 0 outside
        roi_params – (cx, cy, half_side) compatible with downstream helpers
    """
    h, w = image_bgr.shape[:2]

    # Scale image to fit a reasonable display window
    MAX_DISP_W, MAX_DISP_H = 1400, 900
    scale = min(MAX_DISP_W / w, MAX_DISP_H / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display_base = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    font_scale = max(0.5, min(1.2, disp_w / 1200.0))
    line_spacing = int(30 * font_scale)

    # Loupe magnifier settings
    LOUPE_ZOOM = 4.0
    LOUPE_HALF_SIZE = max(30, int(disp_w * 0.06))

    instructions = [
        "Click the 4 INTERIOR CORNERS of the 1cm x 1cm square ROI (any order).",
        "Use the magnifier for precise placement. Press 'r' to restart.",
        "Press ENTER after all 4 corners are clicked to confirm.",
    ]

    corners = []       # original-image coordinates
    mouse_pos = None

    window_name = "Select ROI – Click 4 corners of the square"

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, corners
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            orig_x = x / scale
            orig_y = y / scale
            corners.append((orig_x, orig_y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.imshow(window_name, display_base)   # initial show so moveWindow works
    _center_window(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        disp = display_base.copy()

        # Instructions
        y_text = int(25 * font_scale)
        for line in instructions:
            cv2.putText(disp, line, (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), 2, cv2.LINE_AA)
            y_text += line_spacing

        # Click counter
        count_color = (0, 255, 0) if len(corners) == 4 else (0, 200, 255)
        cv2.putText(disp, f"Corners clicked: {len(corners)} / 4",
                    (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    count_color, 2, cv2.LINE_AA)

        # Draw clicked corners
        for i, (ox, oy) in enumerate(corners):
            dx = int(ox * scale)
            dy = int(oy * scale)
            cv2.drawMarker(disp, (dx, dy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(disp, str(i + 1), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Preview rectangle once all 4 corners are in
        if len(corners) == 4:
            pts_dx = [int(c[0] * scale) for c in corners]
            pts_dy = [int(c[1] * scale) for c in corners]
            rx0d, rx1d = min(pts_dx), max(pts_dx)
            ry0d, ry1d = min(pts_dy), max(pts_dy)
            s_d = max(0, int(shrink * scale))
            cv2.rectangle(disp,
                          (rx0d + s_d, ry0d + s_d),
                          (rx1d - s_d, ry1d - s_d),
                          (255, 0, 0), 2)
            cv2.putText(disp, "Blue = final ROI (shrunk by 5 px). Press ENTER to confirm.",
                        (10, disp_h - 15 - line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 200, 0), 2, cv2.LINE_AA)

        # Magnifier loupe
        if mouse_pos is not None:
            mx, my = mouse_pos
            x0l = max(mx - LOUPE_HALF_SIZE, 0)
            y0l = max(my - LOUPE_HALF_SIZE, 0)
            x1l = min(mx + LOUPE_HALF_SIZE, disp.shape[1] - 1)
            y1l = min(my + LOUPE_HALF_SIZE, disp.shape[0] - 1)
            patch = disp[y0l:y1l, x0l:x1l]
            if patch.size > 0:
                loupe = cv2.resize(
                    patch,
                    (int(patch.shape[1] * LOUPE_ZOOM), int(patch.shape[0] * LOUPE_ZOOM)),
                    interpolation=cv2.INTER_LINEAR,
                )
                lh, lw = loupe.shape[:2]
                # Crosshair
                cv2.line(loupe, (lw // 2, 0), (lw // 2, lh - 1), (0, 0, 0), 3)
                cv2.line(loupe, (0, lh // 2), (lw - 1, lh // 2), (0, 0, 0), 3)
                # Placement
                ox_l = mx + 20
                oy_l = my + 20
                if ox_l + lw > disp.shape[1]:
                    ox_l = mx - lw - 20
                if oy_l + lh > disp.shape[0]:
                    oy_l = my - lh - 20
                ox_l, oy_l = max(0, ox_l), max(0, oy_l)
                # Circular blend mask
                lmask = np.zeros((lh, lw, 3), dtype=np.float32)
                lradius = min(lh, lw) // 2
                cv2.circle(lmask, (lw // 2, lh // 2), lradius, (1.0, 1.0, 1.0), -1)
                sub = disp[oy_l:oy_l + lh, ox_l:ox_l + lw].astype(np.float32)
                blended = sub * (1.0 - lmask) + loupe.astype(np.float32) * lmask
                disp[oy_l:oy_l + lh, ox_l:ox_l + lw] = blended.astype(np.uint8)
                cv2.circle(disp,
                           (ox_l + lw // 2, oy_l + lh // 2),
                           lradius, (255, 255, 255), 1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            corners = []

        if key in (13, 10) and len(corners) == 4:   # Enter
            break

    cv2.destroyWindow(window_name)

    # Build rectangle in original-image coordinates and shrink inward
    orig_xs = [c[0] for c in corners]
    orig_ys = [c[1] for c in corners]

    rx0 = int(round(min(orig_xs))) + shrink
    rx1 = int(round(max(orig_xs))) - shrink
    ry0 = int(round(min(orig_ys))) + shrink
    ry1 = int(round(max(orig_ys))) - shrink

    rx0 = max(0, rx0)
    ry0 = max(0, ry0)
    rx1 = min(w - 1, rx1)
    ry1 = min(h - 1, ry1)

    if rx1 <= rx0 or ry1 <= ry0:
        raise RuntimeError(
            "ROI collapsed after shrinking. Please click the corners more carefully."
        )

    # Build rectangular mask
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    mask_roi[ry0:ry1 + 1, rx0:rx1 + 1] = 255

    cx = (rx0 + rx1) // 2
    cy = (ry0 + ry1) // 2
    half_side = min(rx1 - rx0, ry1 - ry0) // 2   # used downstream like circle radius

    print(f"  ROI rectangle: ({rx0}, {ry0}) – ({rx1}, {ry1}), "
          f"center ({cx}, {cy}), half-side {half_side} px")

    return mask_roi, (cx, cy, half_side)


# =========================
# ROI RECT HELPERS
# =========================

def roi_to_rect(mask_roi):
    """
    Extract the bounding rectangle from a binary ROI mask.
    Returns (x0, y0, x1, y1) in image coordinates.
    """
    ys, xs = np.where(mask_roi == 255)
    if len(ys) == 0:
        raise RuntimeError("roi_to_rect: empty mask supplied")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def rect_to_roi(x0, y0, x1, y1, img_h, img_w):
    """
    Build (mask_roi, roi_params) from a bounding rectangle.
    Clamps coordinates to image bounds.
    Returns (uint8 mask, (cx, cy, half_side)).
    """
    x0c = max(0, min(img_w - 1, x0))
    y0c = max(0, min(img_h - 1, y0))
    x1c = max(0, min(img_w - 1, x1))
    y1c = max(0, min(img_h - 1, y1))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y0c:y1c + 1, x0c:x1c + 1] = 255
    cx = (x0c + x1c) // 2
    cy = (y0c + y1c) // 2
    half_side = min(x1c - x0c, y1c - y0c) // 2
    return mask, (cx, cy, half_side)


# =========================
# PER-IMAGE ROI NUDGE
# =========================

def interactive_roi_nudge(image_bgr, x0, y0, x1, y1, image_name=""):
    """
    Show a ZOOMED crop centred on the ROI and let the user nudge its position
    to compensate for slight frame-to-frame shifts.

    The view zooms in on the ROI (± 50 % padding on each side) so the edges
    of the box are clearly visible for precise alignment.

    Coarse nudge : w / a / s / d  – up / left / down / right  ±10 px
    Fine nudge   : z / x          – left / right               ±1 px
                   , / .          – up   / down                ±1 px
    r            : reset to the original position passed in
    y            : accept this position for ALL remaining images
    Enter/Space  : confirm for this image and advance

    Returns: (x0, y0, x1, y1, apply_to_all)
        x0/y0/x1/y1  – final rectangle in original-image coordinates
        apply_to_all – True if the user pressed 'y'
    """
    WIN = ("Confirm ROI  |  w/a/s/d=10px  z/x=H±1  ,/.=V±1  "
           "r=reset  y=accept-all  Enter=confirm")
    MAX_W, MAX_H = 1400, 900
    h_img, w_img = image_bgr.shape[:2]

    orig_x0, orig_y0, orig_x1, orig_y1 = x0, y0, x1, y1
    cur_x0, cur_y0, cur_x1, cur_y1 = x0, y0, x1, y1
    bname = os.path.basename(image_name) if image_name else "image"

    def _clamp(cx0, cy0, cx1, cy1):
        rw = cx1 - cx0
        rh = cy1 - cy0
        cx0 = max(0, min(w_img - rw, cx0))
        cy0 = max(0, min(h_img - rh, cy0))
        return cx0, cy0, cx0 + rw, cy0 + rh

    def _render(cx0, cy0, cx1, cy1):
        # ---- zoomed crop centred on the ROI --------------------------------
        roi_w = cx1 - cx0
        roi_h = cy1 - cy0
        # Context padding: 50 % of the larger ROI dimension, min 80 px
        ctx = max(80, int(max(roi_w, roi_h) * 0.5))
        sx0 = max(0, cx0 - ctx)
        sy0 = max(0, cy0 - ctx)
        sx1 = min(w_img, cx1 + ctx)
        sy1 = min(h_img, cy1 + ctx)
        crop = image_bgr[sy0:sy1, sx0:sx1]
        ch, cw = crop.shape[:2]
        if cw == 0 or ch == 0:           # safety fallback
            crop = image_bgr
            sx0, sy0 = 0, 0
            ch, cw = image_bgr.shape[:2]

        # Scale to fit MAX_W × MAX_H (upscaling allowed — that's the zoom)
        scale = min(MAX_W / cw, MAX_H / ch)
        out_w = max(1, int(cw * scale))
        out_h = max(1, int(ch * scale))
        scaled = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox onto a fixed MAX_W × MAX_H canvas
        canvas = np.zeros((MAX_H, MAX_W, 3), dtype=np.uint8)
        ox = (MAX_W - out_w) // 2
        oy = (MAX_H - out_h) // 2
        canvas[oy:oy + out_h, ox:ox + out_w] = scaled

        # ROI rectangle in canvas coordinates
        rx0d = ox + int((cx0 - sx0) * scale)
        ry0d = oy + int((cy0 - sy0) * scale)
        rx1d = ox + int((cx1 - sx0) * scale)
        ry1d = oy + int((cy1 - sy0) * scale)
        cv2.rectangle(canvas, (rx0d, ry0d), (rx1d, ry1d), (255, 0, 0), 2)

        # Text overlay
        dx = cx0 - orig_x0
        dy = cy0 - orig_y0
        lines = [
            f"{bname}  |  offset from baseline: ({dx:+d}, {dy:+d}) px",
            "w/a/s/d=10px  z/x=H-fine  ,/.=V-fine  r=reset  y=accept-all  Enter=confirm",
        ]
        for i, txt in enumerate(lines):
            yy = 28 + i * 26
            cv2.putText(canvas, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(canvas, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return canvas

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, MAX_W, MAX_H)
    cv2.imshow(WIN, _render(cur_x0, cur_y0, cur_x1, cur_y1))
    _center_window(WIN, MAX_W, MAX_H)

    apply_to_all = False
    while True:
        key = cv2.waitKey(0)
        k = key & 0xFFFF

        moved = True
        if k in (13, 10):                    # Enter — confirm this image
            moved = False
            break
        elif k == 32:                        # Space — also confirm
            moved = False
            break
        elif k == ord('y'):                  # accept-all remaining images
            apply_to_all = True
            moved = False
            break
        elif k == ord('r'):                  # reset to original position
            cur_x0, cur_y0, cur_x1, cur_y1 = orig_x0, orig_y0, orig_x1, orig_y1
        elif k == ord('w'):                  # coarse up 10 px
            cur_y0 -= 10; cur_y1 -= 10
        elif k == ord('s'):                  # coarse down 10 px
            cur_y0 += 10; cur_y1 += 10
        elif k == ord('a'):                  # coarse left 10 px
            cur_x0 -= 10; cur_x1 -= 10
        elif k == ord('d'):                  # coarse right 10 px
            cur_x0 += 10; cur_x1 += 10
        elif k == ord('z'):                  # fine left 1 px
            cur_x0 -= 1; cur_x1 -= 1
        elif k == ord('x'):                  # fine right 1 px
            cur_x0 += 1; cur_x1 += 1
        elif k == ord(','):                  # fine up 1 px
            cur_y0 -= 1; cur_y1 -= 1
        elif k == ord('.'):                  # fine down 1 px
            cur_y0 += 1; cur_y1 += 1
        else:
            moved = False

        if moved:
            cur_x0, cur_y0, cur_x1, cur_y1 = _clamp(cur_x0, cur_y0, cur_x1, cur_y1)
        cv2.imshow(WIN, _render(cur_x0, cur_y0, cur_x1, cur_y1))

    cv2.destroyWindow(WIN)
    return cur_x0, cur_y0, cur_x1, cur_y1, apply_to_all


# =========================
# DUST MEASUREMENT
# =========================

def measure_dust(image_bgr, mask_roi, baseline_stats=None, dark_thresh_override=None):
    """
    Compute a dust metric inside the ROI using local contrast:
      - blur the image
      - diff = blur - gray
      - treat only the strongest local-dark specks (top DUST_PERCENTILE) as dust

    Returns:
      dust_fraction, dust_pixels, total_pixels, dust_binary, dust_score
      where dust_score is a per-pixel float32 map indicating dust "darkness" (for visualization).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = (mask_roi == 255)
    total_pixels = np.count_nonzero(roi)

    if total_pixels == 0:
        raise RuntimeError("ROI has zero pixels.")

    if baseline_stats is None:
        # Local contrast: blur then subtract
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.subtract(blur, gray)  # higher where pixel is darker than neighborhood

        diff_roi = diff[roi]

        # Threshold at a percentile of local-contrast values
        thresh = np.percentile(diff_roi, DUST_PERCENTILE)

        _, dust_binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        # Only consider dust inside ROI
        dust_binary[~roi] = 0

        dust_pixels = np.count_nonzero(dust_binary == 255)
        dust_fraction = dust_pixels / float(total_pixels)

        # New: compute dust_score as (diff - thresh), clipped to >=0, float32, and zero outside ROI
        diff_f = diff.astype(np.float32)
        dust_score = diff_f - float(thresh)
        dust_score[dust_score < 0] = 0
        dust_score[~roi] = 0

        return dust_fraction, dust_pixels, total_pixels, dust_binary, dust_score
    else:
        # Baseline-based detection: simple absolute comparison to the baseline mean.
        # No shading-correction blur — every pixel in the ROI is measured against
        # the same fixed reference value (base_mean) so the "yardstick" is identical
        # for every image in the batch.  A perfectly clean image returns 0 % dust.
        base_mean, base_std = baseline_stats

        if base_std < 1e-3:
            base_std = 1.0

        gray_f = gray.astype(np.float32)

        # Positive delta = pixel is darker than the baseline mean.
        delta = base_mean - gray_f

        # Determine the darkness threshold (sigma-based default, or fixed override).
        k = BASELINE_SIGMA_K
        sigma_thresh = k * base_std
        abs_thresh = BASELINE_MIN_ABS_DELTA
        dark_thresh = max(sigma_thresh, abs_thresh)

        # A calibration-derived fixed threshold always wins when supplied.
        if dark_thresh_override is not None:
            dark_thresh = float(dark_thresh_override)

        # Dust = pixels inside ROI that are darker than the fixed threshold.
        # No percentile fallback: a clean image genuinely returns 0 %.
        dust_mask = (delta > dark_thresh) & roi

        dust_binary = np.zeros_like(gray, dtype=np.uint8)
        dust_binary[dust_mask] = 255

        dust_pixels = np.count_nonzero(dust_binary == 255)
        dust_fraction = dust_pixels / float(total_pixels)

        # Visualization score: normalize to a FIXED scale so red intensity is
        # consistent across the batch.  4 × threshold = full red; anything above
        # is clamped.  This prevents a nearly-clean image from looking as red as
        # a very dirty one just because both are normalized to their own maximum.
        delta_pos = delta.copy()
        delta_pos[delta_pos < 0] = 0
        delta_pos[~roi] = 0
        fixed_scale = dark_thresh * 4.0
        if fixed_scale > 0:
            dust_score = np.clip(delta_pos / fixed_scale, 0.0, 1.0).astype(np.float32)
        else:
            dust_score = np.zeros_like(delta_pos, dtype=np.float32)

        return dust_fraction, dust_pixels, total_pixels, dust_binary, dust_score

# =========================
# BASELINE DARK THRESHOLD CALIBRATION
# =========================

def compute_baseline_dark_threshold(baseline_image_bgr, baseline_mask, baseline_stats, percentile=95.0):
    """Derive a fixed darkness threshold from the untreated baseline image.

    Uses the same simple delta = base_mean - gray as measure_dust (no shading
    correction blur).  The 95th-percentile of the positive-delta distribution
    inside the baseline ROI sits close to actual background noise, giving a
    sensitive threshold.  Any sample pixel darker than this threshold is dust.
    """
    gray = cv2.cvtColor(baseline_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base_mean, base_std = baseline_stats

    if base_std < 1e-3:
        base_std = 1.0

    # Simple absolute difference — mirrors measure_dust exactly.
    delta = base_mean - gray
    roi = (baseline_mask == 255)
    delta_roi = delta[roi]

    if delta_roi.size == 0:
        # Fallback: no ROI pixels, return None to force default behavior
        return None

    # Only consider pixels that are darker than the baseline mean
    positive = delta_roi[delta_roi > 0]
    if positive.size < 50:
        # Not enough samples; fall back to using the default formula
        k = BASELINE_SIGMA_K
        sigma_thresh = k * base_std
        abs_thresh = BASELINE_MIN_ABS_DELTA
        return float(max(sigma_thresh, abs_thresh))

    raw_thresh = np.percentile(positive, percentile)

    # Take the MINIMUM of the data-derived threshold and the manual abs_thresh
    # so the manual constant acts as a sensitivity ceiling, not a floor.
    # A hard floor of 0.1 prevents the threshold collapsing to near-zero noise.
    abs_thresh = BASELINE_MIN_ABS_DELTA
    dark_thresh = max(min(raw_thresh, abs_thresh), 0.1)

    return float(dark_thresh)
# =========================
# IMAGE PROCESSING
# =========================

# Helper: Let user pick a baseline region in the untreated sample
def pick_baseline_from_image(image_bgr, mask_roi, window_name="Select baseline (untreated sample)", patch_radius=10):
    """
    Let the user click on one or more CLEAN regions of the UNTREATED sample to define
    a baseline grayscale mean and std inside the ROI. All selected patches are pooled
    together to compute (mean, std).

    Returns:
      (baseline_mean, baseline_std)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Scale instruction font size with image size (larger text)
    min_dim = min(h, w)
    font_scale = min_dim / 300.0
    if font_scale < 1.0:
        font_scale = 1.0
    if font_scale > 3.0:
        font_scale = 3.0
    line_spacing = int(32 * font_scale)

    # Precompute ROI rectangle from the mask so we can draw it every frame
    ys, xs = np.where(mask_roi == 255)
    have_roi = False
    rx0 = ry0 = rx1 = ry1 = 0
    if len(xs) > 0:
        rx0, rx1 = int(xs.min()), int(xs.max())
        ry0, ry1 = int(ys.min()), int(ys.max())
        have_roi = True

    # Define a crop centered on the ROI so the interactive window focuses on it.
    if have_roi:
        roi_w = rx1 - rx0
        roi_h = ry1 - ry0
        pad = max(roi_w, roi_h, 50)  # padding around the ROI
        cx_roi = (rx0 + rx1) // 2
        cy_roi = (ry0 + ry1) // 2
        crop_x0 = max(0, cx_roi - pad)
        crop_y0 = max(0, cy_roi - pad)
        crop_x1 = min(w, cx_roi + pad)
        crop_y1 = min(h, cy_roi + pad)
    else:
        # Fallback: use full image if ROI is not available
        crop_x0, crop_y0 = 0, 0
        crop_x1, crop_y1 = w, h

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0
    if crop_w <= 0 or crop_h <= 0:
        # Safety fallback to full image
        crop_x0, crop_y0 = 0, 0
        crop_x1, crop_y1 = w, h
        crop_w, crop_h = w, h

    # Work on a zoomed copy of the cropped region for easier clicking,
    # but map clicks back into the original image coordinates using crop_x0/crop_y0.
    zoom_factor = 2.0
    cropped = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
    display_base = cv2.resize(
        cropped,
        (int(crop_w * zoom_factor), int(crop_h * zoom_factor)),
        interpolation=cv2.INTER_LINEAR,
    )

    # Loupe (magnifier) settings – dynamic size ~1/3 of display width
    LOUPE_ZOOM = 4.0
    disp_h, disp_w = display_base.shape[:2]
    target_fraction = 0.33  # loupe width ~ 1/3 of display width
    # loupe output width ≈ 2 * LOUPE_HALF_SIZE * LOUPE_ZOOM
    LOUPE_HALF_SIZE = int((disp_w * target_fraction) / (2.0 * LOUPE_ZOOM))
    if LOUPE_HALF_SIZE < 30:
        LOUPE_HALF_SIZE = 30

    # Instructions to overlay
    instructions = [
        "Click 3-5 CLEAN, dust-free spots INSIDE the blue square.",
        "Use the magnifier for precise placement.",
        "Press ENTER when done (or 'q' to cancel and use full ROI).",
    ]

    # Store all clicked baseline points (original image coords)
    baseline_points = []
    clicked_pt = None   # last clicked point (for drawing)
    mouse_pos = None    # in display coordinates (int x, y)

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_pt, mouse_pos, baseline_points
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Map from display coords back to original image coords,
            # accounting for the crop offset around the ROI.
            orig_x = x / zoom_factor + crop_x0
            orig_y = y / zoom_factor + crop_y0
            clicked_pt = (orig_x, orig_y)
            baseline_points.append((orig_x, orig_y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.imshow(window_name, display_base)   # initial show so moveWindow works
    _center_window(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # Start from the zoomed base image each frame
        disp_color = display_base.copy()

        # Draw instructions in the upper-left
        y_text = int(25 * font_scale)
        for line in instructions:
            cv2.putText(
                disp_color,
                line,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            y_text += line_spacing

        # Draw ROI rectangle (scaled to zoomed display, using crop offset)
        if have_roi:
            rx0_d = int(round((rx0 - crop_x0) * zoom_factor))
            ry0_d = int(round((ry0 - crop_y0) * zoom_factor))
            rx1_d = int(round((rx1 - crop_x0) * zoom_factor))
            ry1_d = int(round((ry1 - crop_y0) * zoom_factor))
            cv2.rectangle(disp_color, (rx0_d, ry0_d), (rx1_d, ry1_d), (255, 0, 0), 2)

        # Build an in-window circular magnification loupe near cursor
        if mouse_pos is not None:
            mx, my = mouse_pos
            # Crop a small patch around cursor in display coords
            x0 = max(mx - LOUPE_HALF_SIZE, 0)
            y0 = max(my - LOUPE_HALF_SIZE, 0)
            x1 = min(mx + LOUPE_HALF_SIZE, disp_color.shape[1] - 1)
            y1 = min(my + LOUPE_HALF_SIZE, disp_color.shape[0] - 1)

            roi = disp_color[y0:y1, x0:x1]
            if roi.size > 0:
                loupe = cv2.resize(
                    roi,
                    (int(roi.shape[1] * LOUPE_ZOOM), int(roi.shape[0] * LOUPE_ZOOM)),
                    interpolation=cv2.INTER_LINEAR,
                )
                lh, lw = loupe.shape[:2]

                # Draw crosshair at center of loupe for precise selection
                cx_l = lw // 2
                cy_l = lh // 2
                cv2.line(loupe, (cx_l, 0), (cx_l, lh - 1), (0, 0, 0), 3)
                cv2.line(loupe, (0, cy_l), (lw - 1, cy_l), (0, 0, 0), 3)

                # Choose overlay position near cursor and clamp into window
                ox = mx + 20
                oy = my + 20
                if ox + lw > disp_color.shape[1]:
                    ox = mx - lw - 20
                if oy + lh > disp_color.shape[0]:
                    oy = my - lh - 20
                ox = max(0, ox)
                oy = max(0, oy)

                # Prepare circular mask
                mask = np.zeros((lh, lw, 3), dtype=np.float32)
                radius = min(lh, lw) // 2
                center = (lw // 2, lh // 2)
                cv2.circle(mask, center, radius, (1.0, 1.0, 1.0), -1)

                # Blend loupe into the main display using the circular mask
                sub = disp_color[oy:oy + lh, ox:ox + lw].astype(np.float32)
                loupe_f = loupe.astype(np.float32)
                blended = sub * (1.0 - mask) + loupe_f * mask
                disp_color[oy:oy + lh, ox:ox + lw] = blended.astype(np.uint8)

                # Draw a border circle around the loupe
                cv2.circle(
                    disp_color,
                    (ox + center[0], oy + center[1]),
                    radius,
                    (255, 255, 255),
                    1,
                )

        # Draw all clicked baseline points as green dots in display coords
        for (ox, oy) in baseline_points:
            dx = int(round((ox - crop_x0) * zoom_factor))
            dy = int(round((oy - crop_y0) * zoom_factor))
            cv2.circle(disp_color, (dx, dy), 5, (0, 255, 0), -1)

        cv2.imshow(window_name, disp_color)
        key = cv2.waitKey(20) & 0xFF

        # 'q' aborts and falls back to full-ROI baseline
        if key == ord('q'):
            baseline_points = []
            break

        # ENTER or SPACE accept all selected baseline points (if any)
        if key in (13, 10, 32) and len(baseline_points) > 0:
            break

    cv2.destroyWindow(window_name)

    if not baseline_points:
        # Fallback: use entire ROI if no clicks were registered or user pressed 'q'
        roi_pixels = gray[mask_roi == 255]
    else:
        # Collect pixels from patches around ALL selected baseline points
        collected = []
        for (x_f, y_f) in baseline_points:
            x = int(round(x_f))
            y = int(round(y_f))

            x0 = max(0, x - patch_radius)
            x1 = min(w, x + patch_radius + 1)
            y0 = max(0, y - patch_radius)
            y1 = min(h, y + patch_radius + 1)

            patch = gray[y0:y1, x0:x1]
            patch_mask = mask_roi[y0:y1, x0:x1] == 255
            patch_pixels = patch[patch_mask]
            if patch_pixels.size > 0:
                collected.append(patch_pixels.ravel())

        if collected:
            roi_pixels = np.concatenate(collected, axis=0)
        else:
            # If all patches ended up empty, fallback to full ROI
            roi_pixels = gray[mask_roi == 255]

    baseline_mean = float(np.mean(roi_pixels))
    baseline_std = float(np.std(roi_pixels))

    return baseline_mean, baseline_std


# =========================
# IMAGE PROCESSING
# =========================

def create_cropped_highlight_with_footer(overlay_bgr, circle, sample_name, image_name, spin_step, dust_fraction, timestamp_str=None):
    """
    Crop the dust-highlight image around the ROI and add a footer with metadata.

    Crop rules (in original pixels, based on ROI radius r):
      - Width  = 4x ROI height  => 8 * r
      - Height = 4x ROI height for the image content (centered on ROI)
      - Footer height ≈ ROI height (2 * r), appended at the bottom
      - Total height ≈ 5x ROI height (10 * r)

    The footer text has three lines:
      1) sample name | image name | date stamp
      2) spin speed / step
      3) dust coverage %
    """
    h, w = overlay_bgr.shape[:2]
    cx, cy, r = circle
    r = int(r)

    # --- Crop around ROI ---
    half_w = int(4 * r)  # 8r total width
    half_h = int(4 * r)  # 8r content height

    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)

    crop = overlay_bgr[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]

    # --- Footer region ---
    footer_h = int(2 * r)
    if footer_h < 40:
        footer_h = 40  # minimum readable footer height

    # Create new image: crop on top, footer at bottom (white background)
    new_h = crop_h + footer_h
    combined = np.full((new_h, crop_w, 3), 255, dtype=np.uint8)
    combined[0:crop_h, 0:crop_w] = crop

    # --- Footer text ---
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Reasonable font scale based on ROI radius / image width
    base_scale = max(0.5, min(1.5, r / 80.0))
    thickness = 1

    line1 = f"{sample_name} | {image_name} | {timestamp_str}"
    if spin_step is not None:
        line2 = f"Spin speed / step: {spin_step}"
    else:
        line2 = "Spin speed / step: n/a"
    line3 = f"Dust coverage: {dust_fraction * 100.0:.2f}%"

    lines = [line1, line2, line3]

    # Compute vertical placement
    y_start = crop_h + int(0.2 * footer_h)
    line_spacing = int(0.25 * footer_h)

    # Draw text left-aligned with small margin
    x_text = 10
    y = y_start
    for text in lines:
        cv2.putText(
            combined,
            text,
            (x_text, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            base_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += line_spacing

    return combined


def process_single_image(img_path, out_dir, baseline_stats=None, dark_thresh_override=None, debug=False,
                         sample_name=None, spin_step=None, timestamp_str=None,
                         precomputed_mask_roi=None, precomputed_roi_params=None,
                         rotation_angle=0.0):
    """
    Process a single image:
      - apply the batch rotation angle (set interactively on the baseline)
      - auto-detect square ROI from the colored disc
      - measure dust
      - compute a continuous dust_intensity metric from dust_score
      - save dust-highlight overlay
      - (for NEF) save a JPG preview for HTML display
    """
    # Load image (handles NEF via load_image_any)
    image = load_image_any(img_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    # Apply the batch rotation (angle set interactively from the baseline image)
    image = apply_rotation(image, rotation_angle)

    # Figure out base names / extensions
    base_name = os.path.basename(img_path)
    stem, ext = os.path.splitext(base_name)
    ext_lower = ext.lower()

    # Decide what file the HTML report should use for the "Raw" preview
    # For normal images → use the original file
    # For NEF → create a JPG preview: <stem>_raw.jpg
    raw_display_name = base_name
    if ext_lower == ".nef":
        raw_display_name = f"{stem}_raw.jpg"
        raw_display_path = os.path.join(out_dir, raw_display_name)
        # Save the (straightened) NEF-loaded image as a JPG preview for the report
        cv2.imwrite(raw_display_path, image)

    # Use precomputed ROI if supplied, otherwise auto-detect per image
    if precomputed_mask_roi is not None and precomputed_roi_params is not None:
        mask_roi = precomputed_mask_roi
        circle = precomputed_roi_params
    else:
        mask_roi, circle = find_roi_from_grid(image)
    dust_fraction, dust_pixels, total_pixels, dust_binary, dust_score = measure_dust(
        image,
        mask_roi,
        baseline_stats=baseline_stats,
        dark_thresh_override=dark_thresh_override,
    )

    # Continuous dust intensity metric: average dust_score inside ROI.
    # dust_score is 0..1 for darker-than-baseline pixels, 0 elsewhere.
    roi = (mask_roi == 255)
    dust_intensity = 0.0
    if dust_score is not None:
        ds = dust_score.astype(np.float32)
        roi_vals = ds[roi]
        if roi_vals.size > 0:
            dust_intensity = float(roi_vals.mean())

    # Create highlight image (gray background + gradient red overlay based on dust darkness)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Light denoise just for visualization so the background looks less grainy.
    # Dust detection itself still uses the original image via dust_binary/dust_score.
    gray_smooth = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
    base_bgr = cv2.cvtColor(gray_smooth, cv2.COLOR_GRAY2BGR)

    # Use the continuous darkness score for visualization.
    dust_score_f = dust_score.astype(np.float32)
    dust_score_f[dust_score_f < 0] = 0.0
    dust_score_f[dust_score_f > 1] = 1.0

    # Apply a gentle gamma so very slight darkening = very faint tint,
    # and the darkest specks are strongly red.
    gamma = 1.8
    alpha = np.power(dust_score_f, gamma)[..., np.newaxis]  # shape (H, W, 1)

    base_f = base_bgr.astype(np.float32)
    red_img = np.zeros_like(base_f, dtype=np.float32)
    red_img[:, :, 2] = 255.0

    overlay = (base_f * (1.0 - alpha) + red_img * alpha).astype(np.uint8)

    # Draw ROI boundary on overlay (rectangle for user-guided square ROI)
    ys_roi, xs_roi = np.where(mask_roi == 255)
    if len(xs_roi) > 0:
        cv2.rectangle(overlay,
                      (int(xs_roi.min()), int(ys_roi.min())),
                      (int(xs_roi.max()), int(ys_roi.max())),
                      (255, 0, 0), 2)

    # Crop around ROI and add footer with metadata
    cropped_with_footer = create_cropped_highlight_with_footer(
        overlay,
        circle,
        sample_name if sample_name is not None else "",
        base_name,
        spin_step,
        dust_fraction,
        timestamp_str=timestamp_str,
    )

    # Save highlight image
    out_img = os.path.join(out_dir, f"{stem}_dust_highlight.jpg")
    cv2.imwrite(out_img, cropped_with_footer)

    if debug:
        cv2.imshow("Dust Highlight (debug)", cropped_with_footer)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "image": base_name,               # original file (NEF/JPG/PNG/etc.)
        "raw_display": raw_display_name,  # what the HTML report should <img src="...">
        "dust_fraction": dust_fraction,
        "dust_pixels": dust_pixels,
        "total_pixels": total_pixels,
        "dust_intensity": dust_intensity,
    }

# =========================
# PLOT + HTML REPORT
# =========================

def make_sample_plot(sample_dir, sample_name, results):
    """Create dust vs image index plot with fixed y-scale (0–100%).

    Two curves are shown:
      - Coverage area (% of ROI pixels flagged as dust)
      - Intensity (normalized integrated darkness, scaled to 0–100)
    """
    xs = [i + 1 for i in range(len(results))]
    coverage = [r["dust_fraction"] * 100.0 for r in results]
    intensity = [r.get("dust_intensity", 0.0) * 100.0 for r in results]

    plt.figure()
    plt.plot(xs, coverage, marker="o", label="Coverage area (%)")
    plt.plot(xs, intensity, marker="s", linestyle="--", label="Intensity (normalized)")
    plt.xlabel("Spin step (image index)")
    plt.ylabel("Dust metric (%)")
    plt.title(f"Dust vs Spin Step – {sample_name}")
    plt.ylim(0, 100)  # fixed scale for comparison
    plt.grid(True)
    plt.legend()

    plot_path = os.path.join(sample_dir, f"dust_plot_{sample_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def generate_sample_report(sample_dir, sample_name, results, moved_images):
    """
    Create an HTML report in sample_dir that includes:
      - Dust vs spin-speed plot
      - Table: image name, spin step, dust coverage (%)
      - Raw + dust-highlight images for each step

    results: list of dicts like:
        {
          "image": <original filename>,
          "raw_display": <filename to show as raw>  # for NEF, something_raw.jpg
          "dust_fraction": float,
          "dust_pixels": int,
          "total_pixels": int,
        }
    moved_images: list of full paths to the original images in sample_dir
                  (used to preserve ordering / spin step).
    """
    sample_dir = os.path.abspath(sample_dir)

    # Plot
    plot_path = make_sample_plot(sample_dir, sample_name, results)
    plot_name = os.path.basename(plot_path) if plot_path else None

    # Map image name -> spin step (1,2,3,...) based on moved_images order
    name_to_step = {
        os.path.basename(p): i + 1 for i, p in enumerate(moved_images)
    }

    # Build a quick lookup: image name -> result dict
    results_by_name = {r["image"]: r for r in results}

    # Table rows
    table_rows = []
    for r in results:
        img_name = r["image"]
        step = name_to_step.get(img_name, "")
        dust_pct = r["dust_fraction"] * 100.0
        dust_intensity = r.get("dust_intensity", 0.0) * 100.0
        table_rows.append(
            f"<tr><td>{img_name}</td><td>{step}</td><td>{dust_pct:.2f}%</td><td>{dust_intensity:.2f}%</td></tr>"
        )
    table_html = "\n".join(table_rows)

    # Images section
    images_html_parts = []
    for p in moved_images:
        img_name = os.path.basename(p)
        stem, ext = os.path.splitext(img_name)

        # Look up the raw_display name from results, fall back to original name
        r = results_by_name.get(img_name)
        raw_display = r.get("raw_display", img_name) if r is not None else img_name

        highlight_name = f"{stem}_dust_highlight.jpg"

        images_html_parts.append(
            f"<h3>{img_name}</h3>"
            f"<p>Raw:</p>"
            f"<img src='{raw_display}' "
            f"style='max-width:45%; border:1px solid #ccc; margin-right:1rem;'>"
            f"<p>Dust highlight:</p>"
            f"<img src='{highlight_name}' "
            f"style='max-width:45%; border:1px solid #ccc;'>"
            f"<hr>"
        )
    images_html = "\n".join(images_html_parts)

    # Plot HTML block (in case plot_name is None)
    if plot_name:
        plot_html = (
            f'<img src="{plot_name}" '
            f'style="max-width:600px; border:1px solid #ccc;">'
        )
    else:
        plot_html = "<p><i>Plot not available.</i></p>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dust Report – {sample_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
  th {{ background-color: #f0f0f0; }}
</style>
</head>
<body>
  <h1>Dust Report – {sample_name}</h1>

  <h2>Dust vs Spin Step</h2>
  {plot_html}

  <h2>Measurements</h2>
  <table>
    <thead>
      <tr>
        <th>Image name</th>
        <th>Spin speed (step)</th>
        <th>Dust coverage (%)</th>
        <th>Dust intensity (normalized %)</th>
      </tr>
    </thead>
    <tbody>
      {table_html}
    </tbody>
  </table>

  <h2>Images</h2>
  {images_html}

</body>
</html>
"""

    report_path = os.path.join(sample_dir, f"report_{sample_name}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path



# =========================
# FOLDER PROCESSING
# =========================

def process_folder(folder, sample_name, debug_first=False, baseline_from_last=False, run_timestamp=None):
    """
    Process all images in the given folder as one sample.
    Saves everything into results/<sample_name>/ and returns results list.
    """
    folder = os.path.abspath(folder)

    # Only raw images (no dust_highlight)
    images = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
        and "_dust_highlight" not in f
    )

    if not images:
        raise RuntimeError(f"No images found in {folder} with extensions {IMAGE_EXTS}")

    # Choose baseline image:
    #   - If baseline_from_last is False, use the FIRST image (separate untreated baseline)
    #   - If baseline_from_last is True, use the LAST image (no separate baseline; use clean region on last image)
    if baseline_from_last:
        baseline_fname = images[-1]
        print("No separate untreated baseline image selected – using LAST image in series as baseline.")
    else:
        baseline_fname = images[0]
        print("Using FIRST image in series as a separate untreated baseline.")
    baseline_path = os.path.join(folder, baseline_fname)
    print(f"Baseline reference image: {baseline_fname}")

    baseline_image = load_image_any(baseline_path)

    # Let the user rotate the baseline until grid lines are level.
    # The confirmed angle is applied to every image in the batch.
    print("\nOpening baseline image for rotation adjustment...")
    print("  Use a/d (or arrow keys) to rotate, z/x for fine adjustment,")
    print("  r to reset, Enter to confirm, q to cancel (no rotation).")
    rotation_angle = interactive_rotation(baseline_image)
    print(f"  Rotation angle confirmed: {rotation_angle:+.1f} deg")
    baseline_image = apply_rotation(baseline_image, rotation_angle)

    # Let the user draw the ROI rectangle on the baseline image.
    # The card is fixed for the entire batch, so every sample image reuses
    # this same ROI -- no per-image detection needed.
    print("\nROI selection: click the 4 corners of the Sharpie grid square in the")
    print("  baseline image.  Press 'r' to restart, Enter to confirm.")
    baseline_mask, roi_params = find_roi_user_guided(baseline_image)
    print(f"  ROI centre: ({roi_params[0]}, {roi_params[1]}), half-side: {roi_params[2]}px")
    print("  ROI confirmed. A zoomed nudge window will appear for each image.")
    print("  Use w/a/s/d to shift 10 px, z/x/,/. for ±1 px fine nudge.")
    print("  Press 'y' to lock the current position for all remaining images.")

    # Extract the ROI as a rectangle for per-image nudge.
    # base_roi_* holds the original baseline position for offset reporting.
    cur_roi_x0, cur_roi_y0, cur_roi_x1, cur_roi_y1 = roi_to_rect(baseline_mask)
    base_roi_x0, base_roi_y0 = cur_roi_x0, cur_roi_y0
    apply_roi_to_all = False   # set True when user presses 'a' in nudge window

    # Let user interactively select a clean reference region on the untreated sample
    baseline_stats = pick_baseline_from_image(baseline_image, baseline_mask)
    print(f"Baseline stats (gray mean, std): {baseline_stats[0]:.2f}, {baseline_stats[1]:.2f}")

    # Calibrate a fixed darkness threshold from the baseline image (the cleanest
    # image in the batch — the "yardstick").  Every image in the batch uses the
    # same threshold so sensitivity cannot auto-adjust as images get cleaner.
    print("\nCalibrating fixed darkness threshold from baseline image...")
    baseline_dark_thresh = compute_baseline_dark_threshold(
        baseline_image, baseline_mask, baseline_stats
    )
    if baseline_dark_thresh is not None:
        print(f"  Fixed threshold: {baseline_dark_thresh:.3f} gray levels above baseline mean")
        print("  (applied to every image — a clean image will return 0 % dust)")
    else:
        print("  Warning: threshold calibration failed; falling back to per-image default.")

    out_root = "results"
    os.makedirs(out_root, exist_ok=True)
    sample_dir = os.path.join(out_root, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    results = []
    moved_images = []

    # Create processed_images directory inside the sample_images folder
    processed_dir = os.path.join(folder, 'processed_images')
    os.makedirs(processed_dir, exist_ok=True)

    for i, fname in enumerate(images):
        src_path = os.path.join(folder, fname)
        print(f"Processing {fname}...")

        # Copy raw image to processed_images subfolder before moving original
        processed_copy_path = os.path.join(processed_dir, fname)
        shutil.copy2(src_path, processed_copy_path)

        spin_step = i + 1

        # Load + rotate for the ROI confirmation window.
        # (process_single_image will reload the image independently for the
        #  actual measurement; this load is only for the interactive preview.)
        preview = apply_rotation(load_image_any(src_path), rotation_angle)
        img_h, img_w = preview.shape[:2]

        if not apply_roi_to_all:
            # Show the image with the current ROI box; let user nudge if needed.
            nx0, ny0, nx1, ny1, apply_roi_to_all = interactive_roi_nudge(
                preview,
                cur_roi_x0, cur_roi_y0, cur_roi_x1, cur_roi_y1,
                image_name=fname,
            )
            cur_roi_x0, cur_roi_y0, cur_roi_x1, cur_roi_y1 = nx0, ny0, nx1, ny1
            if apply_roi_to_all:
                print(f"  ROI locked at offset "
                      f"({cur_roi_x0 - base_roi_x0:+d}, "
                      f"{cur_roi_y0 - base_roi_y0:+d}) px "
                      f"for all remaining images.")

        # Build the per-image mask from the (possibly adjusted) rectangle.
        sample_mask, sample_roi_params = rect_to_roi(
            cur_roi_x0, cur_roi_y0, cur_roi_x1, cur_roi_y1, img_h, img_w
        )

        res = process_single_image(
            src_path,
            sample_dir,
            baseline_stats=baseline_stats,
            dark_thresh_override=baseline_dark_thresh,
            debug=(debug_first and i == 0),
            sample_name=sample_name,
            spin_step=spin_step,
            timestamp_str=run_timestamp,
            rotation_angle=rotation_angle,
            precomputed_mask_roi=sample_mask,
            precomputed_roi_params=sample_roi_params,
        )
        results.append(res)

        # Move original into sample_dir
        dest_path = os.path.join(sample_dir, fname)
        shutil.move(src_path, dest_path)
        moved_images.append(dest_path)

    # Build rows with sample + spin_step
    rows = []
    for i, r in enumerate(results):
        step = i + 1  # spin step based on order
        rows.append({
            "sample": sample_name,
            "spin_step": step,
            "image": r["image"],
            "dust_fraction": r["dust_fraction"],
            "dust_pixels": r["dust_pixels"],
            "total_pixels": r["total_pixels"],
            "dust_intensity": r.get("dust_intensity", 0.0),
        })

    # --- Per-sample CSV ---
    csv_path = os.path.join(sample_dir, f"dust_results_{sample_name}.csv")
    fieldnames = ["sample", "spin_step", "image",
                  "dust_fraction", "dust_pixels", "total_pixels", "dust_intensity"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- Master CSV (append) ---
    master_csv = os.path.join(out_root, "master_dust_results.csv")
    master_exists = os.path.exists(master_csv)

    if master_exists:
        try:
            with open(master_csv, "r", newline="") as f_master:
                first_line = f_master.readline()
            if "dust_intensity" not in first_line:
                print("[warning] Existing master_dust_results.csv has no 'dust_intensity' column. "
                      "Consider deleting or archiving it so a new file with the updated header can be created.")
        except Exception as e:
            print(f"[warning] Could not inspect master_dust_results.csv header: {e}")
            
    with open(master_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not master_exists:
            writer.writeheader()
        writer.writerows(rows)

    # Generate HTML report (unchanged, still uses original results)
    report_path = generate_sample_report(sample_dir, sample_name, results, moved_images)

    print(f"\nAll outputs saved in: {sample_dir}")
    print(f"  CSV:   {os.path.basename(csv_path)}")
    print(f"  Plot:  dust_plot_{sample_name}.png")
    print(f"  HTML:  {os.path.basename(report_path)}")
    print(f"Master CSV updated: {os.path.basename(master_csv)}")

    return results



# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("\n=== Dust Analysis Tool ===")
    print(f"Version: {TOOL_VERSION}")

    # Always use 'sample_images' folder next to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "sample_images")

    # Build timestamp once
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_sample_name = f"Sample_{timestamp}"

    # Ask user for sample name via a small popup (if Tkinter is available)
    sample_name = default_sample_name

    if TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()  # hide main window

            prompt = (
                "Enter sample name (e.g. Panel_A_01)\n"
                "Leave blank for automatic name."
            )
            user_input = simpledialog.askstring(
                "Dust Analysis – Sample Name",
                prompt,
                parent=root,
            )

            # Clean and build final sample name
            if user_input:
                # strip spaces, replace internal spaces with underscores
                cleaned = user_input.strip()
                cleaned = re.sub(r"\s+", "_", cleaned)
                # remove characters that are problematic for folders
                cleaned = re.sub(r'[^A-Za-z0-9_\-]', "", cleaned)
                if cleaned:
                    sample_name = f"{cleaned}_{timestamp}"
                else:
                    sample_name = default_sample_name
            else:
                sample_name = default_sample_name

            root.destroy()
        except Exception as e:
            print(f"[warning] Tkinter popup failed: {e}")
            print("          Falling back to automatic sample name.")
            sample_name = default_sample_name
    else:
        sample_name = default_sample_name

    # Decide how to select baseline image:
    #   Yes  -> first image in sample_images is separate untreated baseline
    #   No   -> use last image in series as baseline region (no separate baseline file)
    baseline_from_last = False

    if TK_AVAILABLE:
        try:
            root2 = tk.Tk()
            root2.withdraw()
            baseline_msg = (
                "Is there a separate UNTREATED baseline image in the 'sample_images' folder?\n\n"
                "Yes = use the FIRST image in the series as the untreated baseline.\n"
                "No  = use the LAST image in the series and select a clean region on it as baseline."
            )
            has_separate_baseline = messagebox.askyesno(
                "Baseline Image",
                baseline_msg,
                parent=root2,
            )
            baseline_from_last = not has_separate_baseline
            root2.destroy()
        except Exception as e:
            print(f"[warning] Tkinter baseline prompt failed: {e}")
            print("          Defaulting to using FIRST image as baseline.")
            baseline_from_last = False
    else:
        # No Tkinter available – default to first image as baseline
        baseline_from_last = False

    print("\nRunning analysis in batch mode...")
    print(f"  Folder: {folder}")
    print(f"  Sample: {sample_name}")
    print("  (Drop images into sample_images and run the launcher.)\n")

    process_folder(folder, sample_name, debug_first=False, baseline_from_last=baseline_from_last, run_timestamp=timestamp)

    print("\nDone. You can close this window.")
