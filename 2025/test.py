# doc_fourpoint_perp.py
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# General helpers
# =========================
def latest_best(pattern="runs_paper_seg/*/weights/best.pt"):
    cands = sorted(glob.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No weights found at {pattern}")
    return cands[-1]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def order_corners_tl_tr_br_bl(pts4: np.ndarray) -> np.ndarray:
    """
    Order 4 points as TL, TR, BR, BL by angle around centroid,
    then rotate so index 0 looks like top-left.
    """
    pts = np.asarray(pts4, np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    tl_i = np.argmin(pts[:, 1] + 0.35 * pts[:, 0])
    return np.roll(pts, -tl_i, axis=0)

def polygon_area_signed(poly: np.ndarray) -> float:
    p = np.asarray(poly, np.float32).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def overlay_segmentation(img_bgr, mask_uint8, color=(60, 220, 255), alpha=0.45, outline=(0, 140, 255)):
    vis = img_bgr.copy()
    m = mask_uint8 > 127
    vis[m] = (vis[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, outline, 2)
    return vis

import cv2
import numpy as np

def fill_holes(bin_u8: np.ndarray) -> np.ndarray:
    h, w = bin_u8.shape
    ff = bin_u8.copy()
    cv2.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    inv = cv2.bitwise_not(ff)
    return bin_u8 | inv

import cv2
import numpy as np
import cv2
import numpy as np

def refine_page_mask(img_bgr: np.ndarray, mask_u8: np.ndarray,
                     min_keep_frac=0.02,         # keep comps ≥ 2% of largest
                     min_keep_px=1500,           # or ≥ 1500 px
                     close_frac=0.01,            # close gaps ≈ 1% of min(H,W)
                     open_frac=0.005,            # remove thin spikes ≈ 0.5% of min(H,W)
                     smooth_eps_frac=0.0015,     # minimal simplification
                     use_convex_hull=False) -> np.ndarray:
    """
    Refine the binary page mask while preserving edge details.
    """

    H, W = mask_u8.shape[:2]
    min_side = min(H, W)
    mask = (mask_u8 > 0).astype(np.uint8)

    # --- Keep only large connected components ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask_u8

    areas = [cv2.contourArea(c) for c in cnts]
    max_area = max(areas)
    keep_mask = np.zeros_like(mask)
    for c, a in zip(cnts, areas):
        if a >= min_keep_px and a >= max_area * min_keep_frac:
            cv2.drawContours(keep_mask, [c], -1, 1, -1)
    mask = keep_mask

    # --- Close small gaps (very mild) ---
    close_ks = max(3, int(close_frac * min_side))
    if close_ks % 2 == 0: close_ks += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks)))

    # --- Open to remove tiny spikes ---
    open_ks = max(3, int(open_frac * min_side))
    if open_ks % 2 == 0: open_ks += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks)))

    # --- Smooth contour very lightly ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_smooth = np.zeros_like(mask)
    for c in cnts:
        eps = smooth_eps_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if use_convex_hull:
            approx = cv2.convexHull(approx)
        cv2.drawContours(mask_smooth, [approx], -1, 1, -1)

    return (mask_smooth * 255).astype(np.uint8)

# =========================
# Mask utilities
# =========================
def mask_from_ultralytics_result_in_original_coords(result, instance_index: int, H: int, W: int) -> np.ndarray:
    """
    Build a binary mask in original image coordinates using result.masks.xy polygons.
    """
    mask = np.zeros((H, W), np.uint8)
    polys = result.masks.xy[instance_index]
    if isinstance(polys, (list, tuple)):
        polys_list = [np.round(p).astype(np.int32) for p in polys if p is not None and len(p) >= 3]
    else:
        polys_list = [np.round(polys).astype(np.int32)]
    if polys_list:
        cv2.fillPoly(mask, polys_list, 255)
    return mask

def clean_mask_remove_islands(mask_uint8, keep_frac=0.02, min_area_px=1500, close_ks=5, dilate=1):
    if mask_uint8.dtype != np.uint8:
        mask_uint8 = mask_uint8.astype(np.uint8)
    binm = (mask_uint8 > 127).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        out = binm
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = areas.max() if len(areas) else 0
        keep = np.zeros_like(labels, dtype=np.uint8)
        for cid in range(1, num):
            a = stats[cid, cv2.CC_STAT_AREA]
            if a >= max(int(keep_frac * largest), int(min_area_px)):
                keep[labels == cid] = 255
        out = keep

    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        out = cv2.dilate(out, k, iterations=1)
    return out

def largest_contour(mask_uint8):
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


# =========================
# Quad from nearest corners, then perpendicular outward extension
# =========================
def quad_from_corners_extend_perp(hull_xy: np.ndarray, H: int, W: int,
                                  max_iters: int = 5, eps: float = 1e-3) -> np.ndarray:
    """
    1) pick four hull points nearest to image corners TL, TR, BR, BL
    2) create initial quad
    3) extend each side outward along its outward normal until all hull points are inside
       Extension is strictly perpendicular to the side.
    """
    hull = hull_xy.astype(np.float32)
    img_corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)

    # nearest unique hull points to each image corner
    idxs = []
    used = set()
    for c in img_corners:
        d2 = np.sum((hull - c) ** 2, axis=1)
        order = np.argsort(d2)
        pick = None
        for j in order:
            if j not in used:
                pick = int(j); break
        if pick is None:
            pick = int(order[0])
        used.add(pick)
        idxs.append(pick)

    quad = order_corners_tl_tr_br_bl(hull[idxs])

    # ensure a consistent outward normal direction:
    # for each edge, compute the normal that points away from the quad centroid
    def edge_normal_outward(p1, p2, centroid):
        e = p2 - p1
        L = float(np.linalg.norm(e))
        if L < 1e-6:
            return np.array([0.0, 0.0], np.float32)
        e /= L
        # cw rotate gives one normal
        n = np.array([e[1], -e[0]], np.float32)
        mid = 0.5 * (p1 + p2)
        # if this normal points toward the centroid, flip it so it points outward
        if np.dot(n, centroid - mid) > 0:
            n = -n
        return n

    centroid = quad.mean(axis=0)

    # iterative perpendicular outward extension
    for _ in range(max_iters):
        changed = False
        centroid = quad.mean(axis=0)
        for i in range(4):
            p1 = quad[i]
            p2 = quad[(i + 1) % 4]
            n = edge_normal_outward(p1, p2, centroid)
            if n[0] == 0 and n[1] == 0:
                continue

            # signed distance of points to the edge line
            # edge line: <n, x> = <n, p1>
            s_edge = float(np.dot(n, p1))
            s_hull = hull @ n
            d = s_hull - s_edge   # positive means outside along outward normal
            max_out = float(np.max(d))
            if max_out > eps:
                # extend edge outward by max_out
                shift = n * max_out
                quad[i] = p1 + shift
                quad[(i + 1) % 4] = p2 + shift
                changed = True
        if not changed:
            break

    return order_corners_tl_tr_br_bl(quad)


# =========================
# Crop and warp helpers
# =========================
def polygon_crop_rgba_safe(img_bgr: np.ndarray, poly4_xy: np.ndarray,
                           fallback_rect: tuple[int,int,int,int] | None = None):
    """
    Safe polygon crop:
    - clips polygon to image bounds
    - returns non-empty RGBA
    - if bbox would be empty, falls back to a provided rectangle (x,y,w,h)
    """
    H, W = img_bgr.shape[:2]
    poly = np.array(poly4_xy, dtype=np.float32).reshape(-1, 2)

    # clip to image
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

    # quick area check
    x, y, w, h = cv2.boundingRect(np.round(poly).astype(np.int32))
    if w <= 0 or h <= 0:
        if fallback_rect is not None:
            x, y, w, h = fallback_rect
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            b, g, r = cv2.split(img_bgr[y:y+h, x:x+w])
            a = np.full((h, w), 255, np.uint8)
            return cv2.merge((b, g, r, a))
        return None  # still empty

    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    b, g, r = cv2.split(img_bgr)
    rgba_full = cv2.merge((b, g, r, mask))
    crop = rgba_full[y:y+h, x:x+w]
    if crop.size == 0:
        if fallback_rect is not None:
            x, y, w, h = fallback_rect
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            b, g, r = cv2.split(img_bgr[y:y+h, x:x+w])
            a = np.full((h, w), 255, np.uint8)
            return cv2.merge((b, g, r, a))
        return None
    return crop
def is_valid_quad(quad: np.ndarray, W: int, H: int, min_area: float = 25.0) -> bool:
    if quad is None:
        return False
    q = np.asarray(quad, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False
    # clip and area
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area >= min_area

def fallback_quad_from_hull(hull_xy: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(hull_xy.astype(np.int32))
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_corners_tl_tr_br_bl(box)

def bbox_from_hull(hull_xy: np.ndarray) -> tuple[int,int,int,int]:
    x, y, w, h = cv2.boundingRect(hull_xy.astype(np.int32))
    return int(x), int(y), int(w), int(h)


# replace this whole function with the version below
def warp_quad_to_rect(img_bgr, quad_xy, enforce_aspect=None, min_side=64, mode="cover"):
    """
    Perspective-warp the quadrilateral to a rectangle.

    enforce_aspect: None or float
        If set, force the output width:height to this ratio (e.g., sqrt(2) for A-series landscape).
    min_side: int
        Smallest side length for the output.
    mode: "cover" or "pad"
        "cover": resize the rect so that the enforced aspect uses as much of the quad as possible
                 by taking the dominant dimension from the measured quad and solving the other.
        "pad":   first warp to the natural W,H from the quad, then letterbox-pad to the aspect
                 without additional stretching. ("cover" is usually what you want for page scans.)
    """
    q = np.asarray(quad_xy, np.float32).reshape(4, 2)

    # estimate the natural width and height of the quad
    w1 = np.linalg.norm(q[1] - q[0])
    w2 = np.linalg.norm(q[2] - q[3])
    h1 = np.linalg.norm(q[3] - q[0])
    h2 = np.linalg.norm(q[2] - q[1])
    w_nat = max(w1, w2)
    h_nat = max(h1, h2)

    # choose output size
    if enforce_aspect is None:
        W = int(max(w_nat, min_side) + 0.5)
        H = int(max(h_nat, min_side) + 0.5)
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        Hm = cv2.getPerspectiveTransform(q, dst)
        rect = cv2.warpPerspective(img_bgr, Hm, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        return rect

    # enforce a target width:height ratio
    ar = float(enforce_aspect)

    if mode == "cover":
        # keep the dominant dimension from the quad and solve the other from ar
        # choose the scale that changes dimensions the least
        # try width-led
        Ww = max(int(w_nat + 0.5), min_side)
        Hw = max(int(Ww / ar + 0.5), min_side)
        # try height-led
        Hh = max(int(h_nat + 0.5), min_side)
        Wh = max(int(Hh * ar + 0.5), min_side)
        # pick which pair is closer to the natural quad sizes
        err_wled = abs(Hw - h_nat)
        err_hled = abs(Wh - w_nat)
        if err_wled <= err_hled:
            W, H = Ww, Hw
        else:
            W, H = Wh, Hh

        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        Hm = cv2.getPerspectiveTransform(q, dst)
        rect = cv2.warpPerspective(img_bgr, Hm, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        return rect

    elif mode == "pad":
        # first warp to natural size
        W0 = max(int(w_nat + 0.5), min_side)
        H0 = max(int(h_nat + 0.5), min_side)
        dst0 = np.array([[0, 0], [W0 - 1, 0], [W0 - 1, H0 - 1], [0, H0 - 1]], np.float32)
        Hm0 = cv2.getPerspectiveTransform(q, dst0)
        base = cv2.warpPerspective(img_bgr, Hm0, (W0, H0),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        # letterbox to target aspect without stretching content
        # compute target canvas
        if W0 / H0 >= ar:
            W = W0
            H = max(int(W / ar + 0.5), min_side)
        else:
            H = H0
            W = max(int(H * ar + 0.5), min_side)
        canvas = np.full((H, W, 3), 255, np.uint8)
        x = (W - W0) // 2
        y = (H - H0) // 2
        canvas[y:y+H0, x:x+W0] = base
        return canvas

    else:
        raise ValueError('mode must be "cover" or "pad"')

def process_image(model, img_path: Path, out_dir: Path,
                  imgsz=960, conf=0.25, iou=0.5,
                  extend_iters=6, do_warp=True, rect_only=False):
    # read original image
    img = cv2.imread(str(img_path))
    if img is None:
        print("skip unreadable:", img_path.name)
        return
    H, W = img.shape[:2]
    
    # run YOLO on the file path (Ultralytics handles scaling/letterbox)
    r = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        print("no mask:", img_path.name)
        return

    # choose the highest confidence instance of class 0
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    idx = None; bestc = -1.0
    for i, (c, cf) in enumerate(zip(cls, confs)):
        if c == 0 and cf > bestc:
            idx = i; bestc = cf
    if idx is None:
        print("no class 0:", img_path.name)
        return

    # build raw mask in ORIGINAL coordinates from polygons
    mask_raw = np.zeros((H, W), np.uint8)
    polys = r.masks.xy[idx]
    polys = polys if isinstance(polys, (list, tuple)) else [polys]
    polys = [np.round(p).astype(np.int32) for p in polys if p is not None and len(p) >= 3]
    if polys:
        cv2.fillPoly(mask_raw, polys, 255)

    # clean/refine mask
    m_clean = refine_page_mask(img, mask_raw)

    # save segmentation overlay for debugging
    if not rect_only:
        cv2.imwrite(str(out_dir / f"{img_path.stem}_seg.jpg"), overlay_segmentation(img, m_clean))

    # hull from refined mask
    cnt = largest_contour(m_clean)
    if cnt is None or len(cnt) < 3:
        print("no contour:", img_path.name)
        return
    hull = cv2.convexHull(cnt.astype(np.int32)).reshape(-1, 2).astype(np.float32)

    # initial quad from nearest-to-corners, then extend perpendicular outward
    quad = quad_from_corners_extend_perp(hull, H, W, max_iters=extend_iters, eps=1e-3)

    # ensure quad is usable; if not, fall back
    fallback_box = bbox_from_hull(hull)
    if not is_valid_quad(quad, W, H, min_area=64.0):
        quad = fallback_quad_from_hull(hull)

    # try safe crop
    if not rect_only:
        crop_png = polygon_crop_rgba_safe(img, quad, fallback_rect=fallback_box)
        if crop_png is None or crop_png.size == 0:
            # as a last resort, save the hull bbox region as opaque
            x, y, w, h = fallback_box
            b, g, r = cv2.split(img[y:y+h, x:x+w])
            a = np.full((h, w), 255, np.uint8)
            crop_png = cv2.merge((b, g, r, a))
        cv2.imwrite(str(out_dir / f"{img_path.stem}_crop.png"), crop_png)

    # optional perspective warp to rectangle
    if do_warp:
        # A3 landscape uses width:height = sqrt(2)
        rect = warp_quad_to_rect(img, quad, enforce_aspect=np.sqrt(2.0), min_side=64, mode="cover")

        cv2.imwrite(str(out_dir / f"{img_path.stem}_rect.jpg"), rect)

    print("ok:", img_path.name)


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser("Four-point crop with perpendicular outward extension to contain the whole mask")
    ap.add_argument("--weights", type=str, default="", help="path to best.pt (defaults to latest runs_paper_seg/*/weights)")
    ap.add_argument("--source", type=str, default="yolo/images/train", help="folder or single image")
    ap.add_argument("--out", type=str, default="results", help="output folder")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=6, help="max edge-extension iterations")
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--rect-only", action="store_true", help="save only the rect version (no crop/quad/seg overlays)")
    args = ap.parse_args()

    weights = args.weights or latest_best()
    print("Using weights:", weights)
    model = YOLO(weights)

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    src = Path(args.source)
    paths = [src] if src.is_file() else [p for p in sorted(src.glob("*.*")) if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not paths:
        raise FileNotFoundError(f"No images found in {src}")

    for p in paths:
        process_image(model, p, out_dir,
                      imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                      extend_iters=args.iters, do_warp=not args.no_warp, rect_only=args.rect_only)



if __name__ == "__main__":
    main()
