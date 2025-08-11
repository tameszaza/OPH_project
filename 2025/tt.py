#!/usr/bin/env python3
# crop_12_by_ratio_borders.py
#
# Crops each image into 12 tiles using proportion-based boundaries:
#   - Top and bottom horizontal bands defined by Y ratios of image height
#   - Top and bottom columns defined by X ratios of image width
#   - Separate vertical boundaries for top and bottom
#
# New: --flat-output option to save every crop from all images into one folder.
#
# Requirements:
#   pip install opencv-python numpy

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

# ============================ CONFIG ============================
CONFIG = {
    # Y ratios for top and bottom bands: [start_ratio, end_ratio]
    # 0.0 = top of image, 1.0 = bottom of image
    "top_band_ratio": [0.00, 0.53],
    "bottom_band_ratio": [0.53, 1.00],

    # X ratios for TOP band (7 values for 6 columns)
    "top_column_ratios": [0.03, 0.19, 0.34, 0.49, 0.64, 0.79, 0.95],

    # X ratios for BOTTOM band (7 values for 6 columns)
    "bottom_column_ratios": [0.035, 0.185, 0.320, 0.485, 0.645, 0.795, 0.955],

    # Optional nudges (in ratio units, applied to right edges)
    "top_column_nudge_ratio":    [0, 0, 0, 0, 0, 0],
    "bottom_column_nudge_ratio": [0, 0, 0, 0, 0, 0],

    # Inner padding inside each tile (ratio of image dimensions)
    "pad_ratio": 0.0,
}
# ===============================================================

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ratio_to_pixels(ratio_list: List[float], total_size: int) -> List[int]:
    """Convert ratio list to integer pixel coordinates."""
    return [int(round(float(r) * total_size)) for r in ratio_list]

def validate_band_edges(img_w: int, img_h: int,
                        band_ratio: List[float],
                        col_ratios: List[float],
                        nudge_ratios: List[float]) -> Tuple[List[int], List[int]]:
    """Convert ratios to pixel coords and validate."""
    # bands
    band_px = ratio_to_pixels(band_ratio, img_h)
    if not (0 <= band_px[0] < band_px[1] <= img_h):
        raise ValueError(f"Invalid band after conversion: {band_px}")

    # columns
    if len(col_ratios) != 7:
        raise ValueError("Column ratio list must have 7 values.")
    col_px = ratio_to_pixels(col_ratios, img_w)

    if len(nudge_ratios) != 6:
        raise ValueError("Nudge ratio list must have 6 values.")
    for i in range(1, 7):
        col_px[i] += int(round(nudge_ratios[i - 1] * img_w))

    # clamp and enforce monotonic
    col_px[0] = max(0, min(img_w, col_px[0]))
    for i in range(1, len(col_px)):
        col_px[i] = max(col_px[i - 1], min(img_w, col_px[i]))
    col_px[-1] = min(col_px[-1], img_w)

    for i in range(6):
        if col_px[i + 1] <= col_px[i]:
            raise ValueError(f"Degenerate col between {col_px[i]} and {col_px[i+1]}.")

    return band_px, col_px

def draw_preview(img: np.ndarray,
                 top_band: List[int], top_edges: List[int],
                 bottom_band: List[int], bottom_edges: List[int]) -> np.ndarray:
    vis = img.copy()
    h, w = img.shape[:2]

    # horizontal lines
    for y in [top_band[0], top_band[1], bottom_band[0], bottom_band[1]]:
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 2)

    # vertical lines top band
    for x in top_edges[1:-1]:
        cv2.line(vis, (x, top_band[0]), (x, top_band[1]), (0, 255, 255), 2)
    # vertical lines bottom band
    for x in bottom_edges[1:-1]:
        cv2.line(vis, (x, bottom_band[0]), (x, bottom_band[1]), (0, 255, 255), 2)

    # labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 1600.0)
    thick = 2
    yc_top = (top_band[0] + top_band[1]) // 2
    for c in range(6):
        xc = (top_edges[c] + top_edges[c + 1]) // 2
        cv2.putText(vis, f"R1C{c+1}", (xc - 40, yc_top), font, scale, (0, 0, 255), thick)
    yc_bot = (bottom_band[0] + bottom_band[1]) // 2
    for c in range(6):
        xc = (bottom_edges[c] + bottom_edges[c + 1]) // 2
        cv2.putText(vis, f"R2C{c+1}", (xc - 40, yc_bot), font, scale, (0, 0, 255), thick)
    return vis

def list_images(folder: Union[str, Path]):
    ok_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in ok_ext])

def crop_and_save(img_path: Path, out_dir: Path, cfg: dict, flat: bool):
    """
    If flat is True:
      - save all files directly into out_dir with source-stem prefixes
    Else:
      - save into subfolder out_dir/<stem>_tiles
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return
    h, w = img.shape[:2]

    # figure target folder and prefixes
    if flat:
        target_dir = ensure_dir(out_dir)
        prefix = img_path.stem + "_"
    else:
        target_dir = ensure_dir(out_dir / f"{img_path.stem}_tiles")
        prefix = ""  # inside its own folder, no need to prefix

    top_band, top_edges = validate_band_edges(w, h,
                                              cfg["top_band_ratio"],
                                              cfg["top_column_ratios"],
                                              cfg["top_column_nudge_ratio"])
    bottom_band, bottom_edges = validate_band_edges(w, h,
                                                    cfg["bottom_band_ratio"],
                                                    cfg["bottom_column_ratios"],
                                                    cfg["bottom_column_nudge_ratio"])
    pad_x = int(round(cfg["pad_ratio"] * w))
    pad_y = int(round(cfg["pad_ratio"] * h))

    # save references
    cv2.imwrite(str(target_dir / f"{prefix}original.jpg"), img)
    cv2.imwrite(str(target_dir / f"{prefix}grid_preview.jpg"),
                draw_preview(img, top_band, top_edges, bottom_band, bottom_edges))

    tiles_meta = []

    # top row
    for c in range(6):
        x0, x1 = top_edges[c], top_edges[c + 1]
        xx0, xx1 = max(x0 + pad_x, x0), min(x1 - pad_x, x1)
        yy0, yy1 = max(top_band[0] + pad_y, top_band[0]), min(top_band[1] - pad_y, top_band[1])
        tile = img[yy0:yy1, xx0:xx1]
        fname = f"{prefix}R1C{c+1}.jpg"
        cv2.imwrite(str(target_dir / fname), tile)
        tiles_meta.append({"row": 1, "col": c+1, "band_y": top_band, "col_x": [x0, x1], "file": fname})

    # bottom row
    for c in range(6):
        x0, x1 = bottom_edges[c], bottom_edges[c + 1]
        xx0, xx1 = max(x0 + pad_x, x0), min(x1 - pad_x, x1)
        yy0, yy1 = max(bottom_band[0] + pad_y, bottom_band[0]), min(bottom_band[1] - pad_y, bottom_band[1])
        tile = img[yy0:yy1, xx0:xx1]
        fname = f"{prefix}R2C{c+1}.jpg"
        cv2.imwrite(str(target_dir / fname), tile)
        tiles_meta.append({"row": 2, "col": c+1, "band_y": bottom_band, "col_x": [x0, x1], "file": fname})

    # save JSON index
    with open(target_dir / f"{prefix}tiles_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": img_path.name,
            "top_band_px": top_band,
            "bottom_band_px": bottom_band,
            "top_column_edges_px": top_edges,
            "bottom_column_edges_px": bottom_edges,
            "pad_px": [pad_x, pad_y],
            "flat_output": flat,
            "tiles": tiles_meta
        }, f, indent=2, ensure_ascii=False)

def process_folder(in_dir: Union[str, Path], out_root: Union[str, Path], cfg: dict, flat: bool):
    imgs = list_images(in_dir)
    if not imgs:
        print(f"[WARN] No images found under {in_dir}")
        return
    print(f"[INFO] Found {len(imgs)} images.")
    out_root = ensure_dir(out_root)
    for img_path in imgs:
        rel = Path(img_path).relative_to(in_dir)  # not used in flat mode, but kept for symmetry
        crop_and_save(Path(img_path), out_root, cfg, flat)
    print("[INFO] Done.")

def main():
    ap = argparse.ArgumentParser(description="Crop images into 12 tiles using proportion-based boundaries.")
    ap.add_argument("input_dir", help="Folder containing images")
    ap.add_argument("-o", "--out-dir", default="tiles_out", help="Output root folder")
    ap.add_argument("--flat-output", action="store_true",
                    help="Save all crops from all images into the same folder with source-stem prefixes")
    args = ap.parse_args()
    process_folder(Path(args.input_dir), Path(args.out_dir), CONFIG, args.flat_output)

if __name__ == "__main__":
    main()
