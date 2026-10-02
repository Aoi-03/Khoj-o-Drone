#!/usr/bin/env python3
"""
Task 1A – Khojo Drone Survivor Detection Pipeline
=================================================
Pipeline:
  1. Detect ArUco corner markers (IDs 80, 85, 90, 95)
  2. Perspective-transform the arena to 900×900 px
  3. Build a 12×12 grid coordinate system
  4. Name every intersection  A1–K11
  5. Detect red (critical) and yellow (stable) survivors
  6. Reduce each survivor blob to its centroid
  7. Assign nearest grid intersection → write results file
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_IDS   = {80, 85, 90, 95}
CANVAS_SIZE    = 900          # output canvas is CANVAS_SIZE × CANVAS_SIZE px
GRID_DIVISIONS = 12           # 12×12 cells  →  11×11 interior intersections
NUM_LINES      = GRID_DIVISIONS - 1   # 11 interior lines per axis
COLS           = list("ABCDEFGHIJK")  # A … K  (11 columns)
ROWS           = list(range(1, 12))   # 1 … 11 (11 rows)

# ArUco marker → which of its four corners is the *inner* (field) corner
# Corner order per marker: [TL, TR, BR, BL]  (OpenCV convention)
#
#   ID 80  → top-left  marker  → inner corner = BR (index 2)
#   ID 85  → top-right marker  → inner corner = BL (index 3)
#   ID 90  → bottom-right      → inner corner = TL (index 0)
#   ID 95  → bottom-left       → inner corner = TR (index 1)
#
MARKER_INNER_CORNER = {
    80: 2,   # BR
    85: 3,   # BL
    90: 0,   # TL
    95: 1,   # TR
}

# Destination corners on the 900×900 canvas (same order as above: TL,TR,BR,BL)
DEST_CORNERS = {
    80: np.array([0,             0],            dtype=np.float32),  # TL
    85: np.array([CANVAS_SIZE,   0],            dtype=np.float32),  # TR
    90: np.array([CANVAS_SIZE,   CANVAS_SIZE],  dtype=np.float32),  # BR
    95: np.array([0,             CANVAS_SIZE],  dtype=np.float32),  # BL
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – ArUco detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_aruco(image):
    """Return {id: corners_array(4×2)} for markers found in *image*.

    Supports both OpenCV 4.x and OpenCV 5.x ArUco APIs automatically.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    # OpenCV 5+: DetectorParameters() + ArucoDetector
    # OpenCV 4.x: DetectorParameters_create() + detectMarkers directly
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params   = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=params)

    if ids is None:
        return {}

    detected = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        # corners shape: (1, 4, 2) → squeeze to (4, 2)
        detected[int(marker_id)] = marker_corners.reshape(4, 2)

    return detected


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Perspective transform
# ──────────────────────────────────────────────────────────────────────────────

def perspective_transform(image, marker_map):
    """Warp the arena to a CANVAS_SIZE × CANVAS_SIZE top-down view."""
    src_pts  = []
    dst_pts  = []

    for mid in sorted(REQUIRED_IDS):          # consistent ordering
        corners      = marker_map[mid]        # shape (4, 2)
        inner_idx    = MARKER_INNER_CORNER[mid]
        src_pts.append(corners[inner_idx])
        dst_pts.append(DEST_CORNERS[mid])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    M       = cv2.getPerspectiveTransform(src, dst)
    warped  = cv2.warpPerspective(image, M,
                                  (CANVAS_SIZE, CANVAS_SIZE),
                                  flags=cv2.INTER_LINEAR)
    return warped


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 & 4 – Grid coordinate system + intersection names
# ──────────────────────────────────────────────────────────────────────────────

def build_grid():
    """Return pixel coordinates and names for all 11×11 intersections.

    Grid is exactly CANVAS_SIZE / GRID_DIVISIONS per cell.
    Intersections sit at multiples of cell_size: 1×cell … 11×cell.

    Returns
    -------
    grid_pts  : np.ndarray shape (11, 11, 2) – pixel (x, y) per intersection
    names     : list of list of str          – names[row][col]  e.g. "C2"
    cell_size : float
    """
    cell = CANVAS_SIZE / GRID_DIVISIONS       # 75.0 px for 900/12

    # xs and ys for the 11 interior lines
    xs = np.array([(c + 1) * cell for c in range(NUM_LINES)], dtype=np.float32)
    ys = np.array([(r + 1) * cell for r in range(NUM_LINES)], dtype=np.float32)

    # grid_pts[row, col] = (x, y)
    grid_pts = np.zeros((NUM_LINES, NUM_LINES, 2), dtype=np.float32)
    for r in range(NUM_LINES):
        for c in range(NUM_LINES):
            grid_pts[r, c] = (xs[c], ys[r])

    # names[row][col] = e.g. "C2"
    names = []
    for r in range(NUM_LINES):
        row_names = []
        for c in range(NUM_LINES):
            row_names.append(f"{COLS[c]}{ROWS[r]}")
        names.append(row_names)

    return grid_pts, names, cell


def draw_grid(canvas, cell_size):
    """Draw the 11 interior grid lines on *canvas* in bright green."""
    out = canvas.copy()
    for i in range(1, GRID_DIVISIONS):
        pos = int(round(i * cell_size))
        cv2.line(out, (pos, 0),           (pos, CANVAS_SIZE), (0, 255, 0), 1)
        cv2.line(out, (0,   pos),         (CANVAS_SIZE, pos), (0, 255, 0), 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Survivor detection
# ──────────────────────────────────────────────────────────────────────────────

def _largest_contours(mask, min_area=150):
    """Return contours from *mask* whose area ≥ min_area."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]


def detect_survivors(warped):
    """Return (critical_contours, stable_contours) in HSV colour space.

    Critical  = red triangles   (hue ≈ 0 / 180 in HSV)
    Stable    = yellow circles  (hue ≈ 25–35)

    We work in HSV for robust colour isolation.
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # ── Red: two HSV ranges (hue wraps around 0/180) ──────────────────────
    red_lo1 = np.array([0,   120,  80], dtype=np.uint8)
    red_hi1 = np.array([8,   255, 255], dtype=np.uint8)
    red_lo2 = np.array([172, 120,  80], dtype=np.uint8)
    red_hi2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lo1, red_hi1),
        cv2.inRange(hsv, red_lo2, red_hi2),
    )

    # ── Yellow: single HSV range ───────────────────────────────────────────
    yel_lo = np.array([20,  120,  80], dtype=np.uint8)
    yel_hi = np.array([38,  255, 255], dtype=np.uint8)
    mask_yel = cv2.inRange(hsv, yel_lo, yel_hi)

    # Morphological clean-up: close small gaps, remove tiny specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,  kernel)
    mask_yel = cv2.morphologyEx(mask_yel, cv2.MORPH_CLOSE, kernel)
    mask_yel = cv2.morphologyEx(mask_yel, cv2.MORPH_OPEN,  kernel)

    critical_cnts = _largest_contours(mask_red)
    stable_cnts   = _largest_contours(mask_yel)

    return critical_cnts, stable_cnts


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 – Reduce contour → centroid
# ──────────────────────────────────────────────────────────────────────────────

def centroid(contour):
    """Return (x, y) centroid for a contour; falls back to bounding-box centre
    if the moment area is zero."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2, y + h / 2
    return float(cx), float(cy)


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 – Nearest intersection
# ──────────────────────────────────────────────────────────────────────────────

def nearest_intersection(cx, cy, grid_pts, names):
    """Return the name of the intersection closest to (cx, cy)."""
    best_name = None
    best_dist = float("inf")
    for r in range(NUM_LINES):
        for c in range(NUM_LINES):
            gx, gy = grid_pts[r, c]
            dist = (cx - gx) ** 2 + (cy - gy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = names[r][c]
    return best_name


# ──────────────────────────────────────────────────────────────────────────────
# Debug / visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_debug_composite(warped, grid_pts, names, cell_size,
                          critical_cnts, stable_cnts,
                          critical_labels, stable_labels,
                          out_path):
    """Save one composite debug image with grid, contours, centres, labels."""
    vis = draw_grid(warped, cell_size)

    # Draw survivor contours
    cv2.drawContours(vis, critical_cnts, -1, (0, 0, 255),  2)
    cv2.drawContours(vis, stable_cnts,   -1, (0, 255, 255), 2)

    # Draw centroids and labels for critical survivors
    for cnt, label in zip(critical_cnts, critical_labels):
        cx, cy = centroid(cnt)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(vis, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Draw centroids and labels for stable survivors
    for cnt, label in zip(stable_cnts, stable_labels):
        cx, cy = centroid(cnt)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        cv2.putText(vis, label, (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Label all intersection names (small, white)
    for r in range(NUM_LINES):
        for c in range(NUM_LINES):
            gx, gy = grid_pts[r, c]
            cv2.putText(vis, names[r][c],
                        (int(gx) - 10, int(gy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1)

    cv2.imwrite(out_path, vis)
    print(f"[debug] composite image saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Khojo-Drone Task 1A – survivor detection pipeline"
    )
    parser.add_argument("--image", required=True,
                        help="Path to the arena photograph (e.g. image_1.jpg)")
    args = parser.parse_args()

    image_path = args.image

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Loaded {image_path}  ({image.shape[1]}×{image.shape[0]} px)")

    # ── Step 1: ArUco detection ───────────────────────────────────────────────
    marker_map = detect_aruco(image)
    found_ids  = sorted(marker_map.keys())
    print(f"[step 1] Detected marker IDs: {found_ids}")

    missing = REQUIRED_IDS - set(found_ids)
    if missing:
        print(f"[ERROR] Missing required markers: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Perspective transform ─────────────────────────────────────────
    warped = perspective_transform(image, marker_map)
    print(f"[step 2] Warped arena to {warped.shape[1]}×{warped.shape[0]} px")

    # ── Step 3 & 4: Grid + intersection names ─────────────────────────────────
    grid_pts, names, cell_size = build_grid()
    print(f"[step 3/4] Grid built: cell={cell_size:.1f} px, "
          f"intersections A1…K11")

    # ── Step 5: Survivor detection ────────────────────────────────────────────
    critical_cnts, stable_cnts = detect_survivors(warped)
    print(f"[step 5] Found {len(critical_cnts)} critical blob(s), "
          f"{len(stable_cnts)} stable blob(s)")

    # ── Step 6 & 7: Centroids + nearest intersection ──────────────────────────
    critical_labels = []
    for cnt in critical_cnts:
        cx, cy = centroid(cnt)
        label  = nearest_intersection(cx, cy, grid_pts, names)
        critical_labels.append(label)
        print(f"  [critical] centre=({cx:.1f},{cy:.1f}) → {label}")

    stable_labels = []
    for cnt in stable_cnts:
        cx, cy = centroid(cnt)
        label  = nearest_intersection(cx, cy, grid_pts, names)
        stable_labels.append(label)
        print(f"  [stable]   centre=({cx:.1f},{cy:.1f}) → {label}")

    # ── Debug composite image ─────────────────────────────────────────────────
    base        = os.path.splitext(image_path)[0]
    debug_path  = base + "_debug.jpg"
    save_debug_composite(warped, grid_pts, names, cell_size,
                         critical_cnts, stable_cnts,
                         critical_labels, stable_labels,
                         debug_path)

    # ── Write results file ─────────────────────────────────────────────────────
    results_path = base + "_results.txt"
    with open(results_path, "w") as f:
        # Line 1
        ids_str = "[" + ", ".join(str(i) for i in found_ids) + "]"
        f.write(f"Detected marker IDs: {ids_str}\n")
        # Line 2 – blank
        f.write("\n")
        # Line 3
        crit_str = ", ".join(critical_labels)
        f.write(f"Critical Survivors: {crit_str}\n")
        # Line 4
        stab_str = ", ".join(stable_labels)
        f.write(f"Stable Survivors: {stab_str}\n")

    print(f"[done] Results written → {results_path}")
    print("--- Results Preview ---")
    with open(results_path) as f:
        print(f.read(), end="")


if __name__ == "__main__":
    main()
