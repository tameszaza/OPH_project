# api.py
# Flask API for subject enrollment scanning
# Pipeline:
#   1) YOLO segmentation isolates the document and crops it
#   2) DocScanner dewarps the cropped doc IN-MEMORY (no repo I/O); fallback to crop if unavailable
#   3) Ratio-based grid cropping splits the rectified doc into 12 tiles (R1C1..R2C6)
#   4) Per-tile detection with your best model to pair sticker colors with nearest codes
#   5) Validate and append a local CSV record
#   6) Recommend a mascot + dynamic Thai description (LLM or rule-based)
#
# Env for LLM (optional):
#   export GEMMA_API_URL="https://<your-gemma-endpoint>"
#   export GEMMA_API_KEY="<your-key>"
#   export GEMMA_MODEL="gemma-3-27b-instruct"
#
# Env for DocScanner weights (optional; falls back to ./DocScanner/model_pretrained/* if present):
#   export DOCSCANNER_SEG_WEIGHTS="/absolute/path/to/u2netp.pth"
#   export DOCSCANNER_REC_WEIGHTS="/absolute/path/to/docscanner.pth"

from flask import Flask, request, jsonify, render_template
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import socket
import platform
import warnings
warnings.filterwarnings('ignore')

# ===== import your two helper modules =====
# Make sure these two files sit next to this api.py, or are importable on PYTHONPATH
import test as docseg              # your segmentation and rectification utilities (only used for latest_best fallback)
import tt as colcrop               # your 12-tile ratio cropper

# ---- LLM mascot recommender (dynamic per record) ----
import re, json, requests
from collections import Counter

# ---- Torch for DocScanner in-memory dewarp ----
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

PROVIDER = "vertex"
GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMMA_MODEL   = "gemini-2.0-flash"
GEMMA_API_KEY = "replaceThisWithUr" # not used directly by Vertex REST, ok to keep
LLM_DEBUG     = True

# -------------------- constants and folders --------------------
UPLOAD_FOLDER = 'uploads'
SEG_OUT_DIR = 'seg_nofixed'  # absolute path per requirement
TILES_OUT_DIR = 'tiles_out'
PROCESSED_FOLDER = 'processed'
CSV_PATH = 'enrollment_results.csv'

for d in [UPLOAD_FOLDER, SEG_OUT_DIR, TILES_OUT_DIR, PROCESSED_FOLDER]:
    os.makedirs(d, exist_ok=True)

# ------------- device and host info for CSV meta ---------------
hostname = socket.gethostname()
system_name = platform.system()
node_name = platform.node()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

# ------------------ subject and indexing data ------------------
index_table = (0, 9, 17, 25, 36, 46, 56, 67, 75, 85, 90, 98)

class data:
    def __init__(self, stem_count, foreign_count, total, blue_count, yellow_count, red_count, brown_count, subject_list):
        self.stem_count = stem_count
        self.foreign_count = foreign_count
        self.total = total
        self.blue_count = blue_count
        self.yellow_count = yellow_count
        self.red_count = red_count
        self.brown_count = brown_count
        self.subject_list = subject_list

subjects = (
    ("ค30203", "หลักคณิตศาสตร์"),
    ("ค30204", "สถิติ"),
    ("ค30205", "แคลคูลัสขั้นสูง"),
    ("ค30206", "ทฤษฎีกราฟ"),
    ("ค30207", "กระบวนการสโตแคสติก"),
    ("ค30208", "พีชคณิตเชิงเส้นเบื้องต้น"),
    ("ค30209", "การวิเคราะห์เชิงจริงเบื้องต้น"),
    ("ค30210", "สมการเชิงอนุพันธ์สามัญ"),
    ("ค30211", "การหาค่าเหมาะสมและการเรียนรู้แบบเสริมกำลัง"),
    ("ค30212", "คณิตศาสตร์เชิงคำนวณและการสร้างแบบจำลอง"),
    ("ค30213", "หัวข้อพิเศษทางคณิตศาสตร์"),
    ("ว30301", "กลศาสตร์คลาสสิก"),
    ("ว30302", "ทัศนศาสตร์ประยุกต์"),
    ("ว30303", "ฟิสิกส์ดาราศาสตร์เบื้องต้น"),
    ("ว30304", "ทฤษฎีสนามคลาสสิกและฟิสิกส์อนุภาคเบื้องต้น"),
    ("ว30305", "กลศาสตร์สถิติ"),
    ("ว30306", "กลศาสตร์ฐานแคลคูลัส"),
    ("ว30307", "ไฟฟ้าและแม่เหล็กฐานแคลคูลัส"),
    ("ว30308", "หัวข้อพิเศษทางฟิสิกส์"),
    ("ว30401", "เคมีในชีวิตประจำวัน"),
    ("ว30402", "แนวหน้าของวัสดุศาสตร์ชั้นนำ"),
    ("ว30403", "เคมีวิเคราะห์เชิงไฟฟ้าเบื้องต้น"),
    ("ว30404", "เทคนิคเคมีวิเคราะห์ขั้นสูง"),
    ("ว30405", "สมบัติของสสารขั้นสูง"),
    ("ว30406", "เคมีอนินทรีย์ จลนศาสตร์เคมีขั้นสูง และอุณหพลศาสตร์เคมี"),
    ("ว30407", "เคมีอินทรีย์ขั้นสูง"),
    ("ว30408", "กลศาสตร์ควอนตัมเชิงโมเลกุลเบื้องต้นและสเปกโทรสโกปี"),
    ("ว30409", "การวิเคราะห์และเทคนิคทางชีวเคมี"),
    ("ว30410", "หัวข้อพิเศษทางเคมี"),
    ("ว30501", "ปฏิบัติการชีววิทยาสังเคราะห์ขั้นพื้นฐาน"),
    ("ว30502", "ปฏิบัติการชีววิทยาของพืชและสัตว์ขั้นสูง"),
    ("ว30503", "เทคนิคปฏิบัติการทางจุลชีววิทยา"),
    ("ว30504", "เทคโนโลยีชีวภาพ"),
    ("ว30505", "พันธุศาสตร์ขั้นสูงและชีววิทยาโมเลกุล"),
    ("ว30506", "ประสาทชีววิทยา"),
    ("ว30507", "ชีววิทยาของมะเร็ง"),
    ("ว30508", "ชีววิทยาเพื่อสิ่งแวดล้อมที่ยั่งยืน"),
    ("ว30509", "นิเวศวิทยาภาคสนาม"),
    ("ว30510", "หัวข้อพิเศษทางชีววิทยา"),
    ("ว30601", "บรรพชีวินวิทยา"),
    ("ว30602", "อัญมณีวิทยา"),
    ("ว30603", "หัวข้อพิเศษทางวิทยาศาสตร์โลก"),
    ("ว30604", "หัวข้อพิเศษทางดาราศาสตร์"),
    ("ว30703", "การเขียนโปรแกรมคอมพิวเตอร์เบื้องต้น 2"),
    ("ว30704", "โครงสร้างข้อมูลและอัลกอริทึมเบื้องต้น"),
    ("ว30705", "คอมพิวเตอร์ออกแบบและสร้างต้นแบบ 1"),
    ("ว30706", "คอมพิวเตอร์ออกแบบและสร้างต้นแบบ 2"),
    ("ว30707", "การเขียนโปรแกรมภาษาไพทอน"),
    ("ว30708", "วิทยาการคอมพิวเตอร์ทั่วไป 1"),
    ("ว30709", "วิทยาการคอมพิวเตอร์ทั่วไป 2"),
    ("ว30710", "การเรียนรู้ของเครื่อง 1"),
    ("ว30711", "การเรียนรู้ของเครื่อง 2"),
    ("ว30712", "ระบบปฏิบัติการหุ่นยนต์และการประยุกต์"),
    ("ว30713", "หัวข้อพิเศษทางวิทยาการคอมพิวเตอร์"),
    ("ว30807", "พื้นฐานการคำนวณและการจำลองเชิงวิทยาศาสตร์"),
    ("ว30808", "ฟิสิกส์เชิงคำนวณเบื้องต้น"),
    ("ว30809", "ฟิสิกส์เชิงคำนวณขั้นสูง"),
    ("ว30810", "เคมีเชิงคำนวณพื้นฐาน"),
    ("ว30811", "ชีวสารสนเทศและชีววิทยาเชิงคำนวณเบื้องต้น"),
    ("ว30812", "คณิตศาสตร์สำหรับฟิสิกส์"),
    ("ว30813", "ชีวฟิสิกส์"),
    ("ว30814", "วิศวกรรมโปรตีนและการนำไปใช้"),
    ("ว30815", "นวัตกรรมเพื่อสิ่งแวดล้อมที่ยั่งยืน"),
    ("ว30816", "ภูมิศาสตร์สิ่งแวดล้อม"),
    ("ว30817", "การท่องเที่ยวอย่างยั่งยืน"),
    ("ว30901", "ปรัชญาและประวัติศาสตร์วิทยาศาสตร์"),
    ("ว30902", "ทรัพย์สินทางปัญญา"),
    ("ท30201", "การเล่าเรื่อง"),
    ("ท30202", "การเขียนสารคดี"),
    ("ท30203", "ความพิศวงในวรรณกรรม"),
    ("ท30204", "รสแห่งวรรณกรรม"),
    ("ท30205", "วรรณกรรมพื้นบ้าน"),
    ("ท30206", "วรรณกรรมวิทยาศาสตร์"),
    ("ท30207", "คติชนวิทยา"),
    ("ท30208", "เอกัตศึกษาภาษาและวรรณกรรม"),
    ("ศ30201", "พื้นฐานการออกแบบ"),
    ("ศ30202", "ศิลปะและเทคโนโลยี"),
    ("ศ30203", "จิตรกรรมสร้างสรรค์"),
    ("ศ30204", "ศิลปะเพื่อสังคม"),
    ("ศ30205", "ทักษะดนตรีไทย"),
    ("ส30201", "จิตวิทยาการดำรงชีวิต"),
    ("ส30202", "มหาศึกชิงบัลลังก์กับสังคมศึกษา"),
    ("ส30203", "เอกัตศึกษา สังคมศาสตร์และมนุษยศาสตร์ 1"),
    ("ส30204", "เอกัตศึกษา สังคมศาสตร์และมนุษยศาสตร์ 2"),
    ("ส30205", "สัมมนาประวัติศาสตร์"),
    ("อ30207", "การพูดในที่ชุมชน"),
    ("อ30208", "วรรณกรรมมีชีวิต"),
    ("อ30209", "การสื่อสารระหว่างวัฒนธรรม"),
    ("อ30210", "การสื่อสารทางวิทยาศาสตร์"),
    ("อ30211", "กลยุทธ์การสอบเพื่อเตรียมตัวสำหรับการสอบแบบทดสอบมาตรฐาน"),
    ("ก30201", "ภาษาเกาหลี 1"),
    ("ก30202", "ภาษาเกาหลี 2"),
    ("จ30201", "ภาษาจีน 1"),
    ("จ30202", "ภาษาจีน 2"),
    ("ซ30201", "ภาษารัสเซีย 1"),
    ("ซ30202", "ภาษารัสเซีย 2"),
    ("ญ30201", "ภาษาญี่ปุ่น 1"),
    ("ญ30202", "ภาษาญี่ปุ่น 2"),
    ("ป30201", "ภาษาสเปน 1"),
    ("ป30202", "ภาษาสเปน 2"),
    ("ฝ30201", "ภาษาฝรั่งเศส 1"),
    ("ฝ30202", "ภาษาฝรั่งเศส 2"),
    ("ย30201", "ภาษาเยอรมัน 1"),
    ("ย30202", "ภาษาเยอรมัน 2"),
    ("ร30201", "ภาษาอาหรับ 1"),
    ("ร30202", "ภาษาอาหรับ 2")
)

# ---------------- YOLO models loaded once ----------------
DOC_SEG_WEIGHTS = 'runs_paper_seg/y11n_baseline4/weights/best.pt'
if DOC_SEG_WEIGHTS and Path(DOC_SEG_WEIGHTS).exists():
    seg_model = YOLO(DOC_SEG_WEIGHTS)
else:
    seg_model = YOLO(docseg.latest_best())

DET_WEIGHTS = 'datasets/detection_partial/runs/yolo11_det/weights/best.pt'
yolo_det_model = YOLO(DET_WEIGHTS)

# ---------------- CSV helpers ----------------
SUBJECT_CODES = [code for code, _ in subjects]
BASE_COLUMNS = [
    'count', 'timestamp', 'is_valid', 'reasons',
    'hostname', 'system_name', 'processor', 'version', 'machine',
    'sem3', 'sem4', 'sem5', 'sem6'
]
CSV_COLUMNS = BASE_COLUMNS + SUBJECT_CODES

def ensure_csv():
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)

def load_csv():
    ensure_csv()
    df = pd.read_csv(CSV_PATH, dtype=str).fillna('')
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = ''
    return df[CSV_COLUMNS].copy()

def save_csv_append(row_dict: dict):
    df = load_csv()
    try:
        last_count = int(df['count'].replace('', '0').astype(int).max()) if len(df) else 0
    except Exception:
        last_count = 0
    row_dict['count'] = last_count + 1
    out = {c: '' for c in CSV_COLUMNS}
    out.update({k: v for k, v in row_dict.items() if k in CSV_COLUMNS})
    df = pd.concat([df, pd.DataFrame([out])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

import cv2
import numpy as np

def stretch_to_ratio(img: np.ndarray, ratio_w: int = 42, ratio_h: int = 30) -> np.ndarray:
    """
    Stretch (non-uniformly resize) an OpenCV image so its aspect ratio becomes ratio_w:ratio_h.
    ยืดด้านที่สั้นกว่าให้พอดีกับสัดส่วน 42:30 โดยคงด้านที่ยาวกว่าไว้เท่าเดิม (ไม่มีการครอป/ขอบดำ)

    Args:
        img: OpenCV image (H x W x C) in BGR or grayscale.
        ratio_w: target ratio width part (default 42).
        ratio_h: target ratio height part (default 30).

    Returns:
        Resized (stretched) image with aspect exactly ratio_w:ratio_h.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    h, w = img.shape[:2]
    target_ar = ratio_w / ratio_h
    curr_ar = w / h

    # Already at target ratio
    if abs(curr_ar - target_ar) < 1e-12:
        return img

    if curr_ar < target_ar:
        # Width is too short relative to height -> stretch width
        new_w = int(round(h * target_ar))
        new_h = h
    else:
        # Height is too short relative to width -> stretch height
        new_w = w
        new_h = int(round(w / target_ar))

    # Pick interpolation suited to the scale direction
    if new_w >= w and new_h >= h:
        interp = cv2.INTER_CUBIC      # upscaling
    elif new_w <= w and new_h <= h:
        interp = cv2.INTER_AREA       # downscaling
    else:
        interp = cv2.INTER_LINEAR     # mixed scale

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------- validation ----------------
def validate_output(output):
    reasons = []
    is_valid = True
    if output.stem_count < 8:
        is_valid = False
        reasons.append('Stem count is less than 8.')
    if output.foreign_count != 2:
        is_valid = False
        reasons.append('Foreign count is not equal to 2.')
    if output.total < 14:
        is_valid = False
        reasons.append('Total count is less than 14.')
    if any([output.blue_count > 4, output.yellow_count > 4, output.red_count > 4, output.brown_count > 4]):
        is_valid = False
        reasons.append('One or more color counts are greater than 4.')
    return is_valid, reasons

def save_annotated_image(image_path: str, detections: list, output_name: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read {image_path} for annotation")
        return
    for color_name, (code, name) in detections:
        cv2.putText(
            img,
            f"{color_name} - {code}",
            (10, 30 + 30 * detections.index([color_name, (code, name)])),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
    save_path = Path(PROCESSED_FOLDER) / output_name
    cv2.imwrite(str(save_path), img)
    print(f"[INFO] Annotated image saved to {save_path}")

# ---------------- tiny utils ----------------
def iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea + 1e-9)

def get_selected_choices(i, img, output: data, filename_for_log: str, save_dir: str = None):
    """
    Runs detection on a single tile, updates output, and optionally saves
    per-tile inference artifacts into save_dir.

    New logic:
      - After collecting 'Code' boxes, compute each box's x-center.
      - Drop codes with |x - mean_x| > 2*std_x (per tile) so only vertically-aligned codes remain.
    """
    model = yolo_det_model
    results = model(img, verbose=False)
    dets = results[0].boxes

    if hasattr(results[0], "names") and isinstance(results[0].names, dict):
        id2name = results[0].names
    else:
        id2name = {0: "Blue", 1: "Yellow", 2: "Red", 3: "Brown", 4: "Code"}

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tile_stem = Path(filename_for_log).stem
        vis = results[0].plot()  # BGR numpy array
        cv2.imwrite(str(Path(save_dir) / f"{tile_stem}_det.jpg"), vis)
        lines = []
        json_list = []
        for b in dets:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf.item())
            cls_id = int(b.cls.item())
            name = id2name.get(cls_id, str(cls_id))
            lines.append(f"{name} {conf:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
            json_list.append({
                "name": name,
                "cls_id": cls_id,
                "conf": conf,
                "bbox_xyxy": [x1, y1, x2, y2]
            })
        (Path(save_dir) / f"{tile_stem}_det.txt").write_text("\n".join(lines), encoding="utf-8")
        import json as _json
        (Path(save_dir) / f"{tile_stem}_det.json").write_text(_json.dumps(json_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- pairing logic with right-side drop ----------
    code_boxes = []      # list of (bbox_xyxy, score)
    sticker_boxes = []   # list of (bbox_xyxy, score, color_name)

    img_area = img.shape[0] * img.shape[1]
    max_area = 0.05 * img_area  # ignore abnormally large "code" boxes in a tile

    for b in dets:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        score = float(b.conf.item())
        cls_id = int(b.cls.item())
        bbox = [x1, y1, x2, y2]
        bbox_area = (x2 - x1) * (y2 - y1)

        if cls_id == 4:  # Code
            if bbox_area <= max_area:
                code_boxes.append((bbox, score))
        else:
            class_names = ['Blue', 'Yellow', 'Red', 'Brown']
            if 0 <= cls_id < len(class_names):
                sticker_boxes.append((bbox, score, class_names[cls_id]))

    # ---- NEW: filter Code boxes by x-center (keep within mean ± 2*std) ----
    if len(code_boxes) >= 3:
        xs = np.array([0.5 * (b[0][0] + b[0][2]) for b in code_boxes], dtype=np.float32)
        mu = float(xs.mean())
        sigma = float(xs.std(ddof=0))
        if sigma > 1e-6:
            keep_mask = np.abs(xs - mu) <= (2.0 * sigma)
            kept = [cb for cb, km in zip(code_boxes, keep_mask.tolist()) if km]
            # Keep at least one if all got filtered due to extreme sigma anomalies
            if kept:
                code_boxes = kept
        # if sigma ~ 0, do nothing (all aligned already)

    # De-duplicate stickers by IOU, preferring higher score
    def _iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return inter / (area_a + area_b - inter + 1e-9)

    filtered = []
    for ii, (box1, s1, c1) in enumerate(sticker_boxes):
        keep = True
        for jj, (box2, s2, c2) in enumerate(sticker_boxes):
            if ii == jj:
                continue
            if _iou(box1, box2) > 0.65 and s1 < s2:
                keep = False
                break
        if keep:
            filtered.append((box1, s1, c1))
    sticker_boxes = filtered

    # Sort remaining codes by their top-y (ascending)
    sorted_codes = sorted(code_boxes, key=lambda x: x[0][1])

    def _center(box):
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    for sticker_bbox, _, color in sticker_boxes:
        if not sorted_codes:
            continue

        sx, sy = _center(sticker_bbox)

        # find nearest code (euclidean center distance)
        nearest_box = None
        min_d = float('inf')
        for code_bbox, _ in sorted_codes:
            cx, cy = _center(code_bbox)
            d = np.hypot(sx - cx, sy - cy)
            if d < min_d:
                min_d = d
                nearest_box = code_bbox

        cx, cy = _center(nearest_box)
        code_w = max(1.0, nearest_box[2] - nearest_box[0])
        x_tolerance = 0.05 * code_w  # 5% of code width

        # require sticker to be left of (or slightly overlapping) the code
        if sx >= cx - x_tolerance:
            continue

        code_only = [c[0] for c in sorted_codes]
        if nearest_box in code_only:
            order = code_only.index(nearest_box) + 1
            output.subject_list.append([color, subjects[index_table[i] + order - 1]])
            output.total += 1

            if color == 'Blue':
                output.blue_count += 1
            elif color == 'Red':
                output.red_count += 1
            elif color == 'Brown':
                output.brown_count += 1
            else:
                output.yellow_count += 1

            if i <= 6:
                output.stem_count += 1
            elif i > 9:
                output.foreign_count += 1

    return output

# ---------------- DocScanner in-memory integration ----------------
BASE_DIR = Path(__file__).resolve().parent
# Accept either "DocScanner" or "DocMatcher" as folder name beside api.py
DOCSCANNER_DIR_CANDIDATES = [BASE_DIR / "DocScanner", BASE_DIR / "DocMatcher"]
DOCSCANNER_DIR = next((p for p in DOCSCANNER_DIR_CANDIDATES if p.exists()), None)

if DOCSCANNER_DIR is not None:
    sys.path.insert(0, str(DOCSCANNER_DIR))
    try:
        from model import DocScanner as _DocScanner
        from seg import U2NETP as _U2NETP
    except Exception as e:
        print(f"[DocScanner] Import failed: {e}")
        DOCSCANNER_DIR = None
else:
    print("[DocScanner] repo not found alongside api.py; dewarp will be skipped.")

DOCSCANNER_SEG_WEIGHTS = os.environ.get(
    "DOCSCANNER_SEG_WEIGHTS",
    str((DOCSCANNER_DIR / "model_pretrained" / "seg.pth")) if DOCSCANNER_DIR else ""
)
DOCSCANNER_REC_WEIGHTS = os.environ.get(
    "DOCSCANNER_REC_WEIGHTS",
    str((DOCSCANNER_DIR / "model_pretrained" / "DocScanner-L.pth")) if DOCSCANNER_DIR else ""
)

_docscanner_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_docscanner_model = None

class _DocScannerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.msk = _U2NETP(3, 1)
        self.bm  = _DocScanner()

    def forward(self, x):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)   # keep exact unpacking
        msk = (msk > 0.5).float()                   # same 0.5 threshold
        x = msk * x
        bm = self.bm(x, iters=12, test_mode=True)
        bm = (2 * (bm / 286.8) - 1) * 0.99          # same scaling
        return bm

# --- add these helpers (verbatim official behavior) ---
def _reload_seg_model_official(model: nn.Module, path: str):
    if not bool(path):
        return model
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location='cpu')
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def _reload_rec_model_official(model: nn.Module, path: str):
    if not bool(path):
        return model
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location='cpu')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def _build_docscanner():
    if DOCSCANNER_DIR is None:
        return None
    try:
        m = _DocScannerNet().to(torch.device('cpu')).eval()  # official assumes CUDA
        _reload_seg_model_official(m.msk, DOCSCANNER_SEG_WEIGHTS)
        _reload_rec_model_official(m.bm,  DOCSCANNER_REC_WEIGHTS)
        return m
    except Exception as e:
        print(f"[DocScanner] Build failed: {e}")
        return None


_docscanner_model = _build_docscanner()

def docscanner_dewarp_numpy(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    In-memory DocScanner: BGR numpy -> BGR numpy (rectified).
    Returns None if DocScanner is unavailable or on error.
    """
    if _docscanner_model is None:
        return None
    try:
        # BGR -> RGB float [0,1]
        im_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w, _ = im_rgb.shape

        # 288x288 as in reference
        im_resized = cv2.resize(im_rgb, (288, 288), interpolation=cv2.INTER_LINEAR)
        im_t = torch.from_numpy(im_resized.transpose(2, 0, 1)).unsqueeze(0).to(_docscanner_device)
        
        with torch.no_grad():
            bm = _docscanner_model(im_t).detach().cpu()  # (1,2,288,288) in [-1,1]

        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))

        grid = np.stack([bm0, bm1], axis=2)  # (h,w,2), values already in [-1,1]
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(_docscanner_device).float()

        src_t = torch.from_numpy(im_rgb.transpose(2, 0, 1)).unsqueeze(0).to(_docscanner_device).float()
        out = F.grid_sample(src_t, grid_t, align_corners=True)  # defaults: bilinear/zeros


        out_rgb = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr
    except Exception as e:
        print(f"[DocScanner] Inference error: {e}")
        return None

# ---------------- segmentation + tiling preprocess ----------------
def preprocess(image_path: str):
    """
    Segment -> CROP (no internal warp) -> DocScanner dewarp in memory (fallback to crop) -> 12 tiles.
    Writes one rectified image into SEG_OUT_DIR for downstream tiler/annotation.
    Returns: (list_of_tiles, rect_path)
    """
    img_path = Path(image_path)
    out_dir = Path(SEG_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original image
    src = cv2.imread(str(img_path))
    if src is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    H, W = src.shape[:2]

    # Run YOLO segmentation to find the document mask/box
    results = seg_model(str(img_path), imgsz=960, conf=0.25, iou=0.5, verbose=False)
    r0 = results[0]

    # Helper: compute bbox from masks (choose the largest mask by area)
    def bbox_from_masks(masks):
        best = None
        best_area = -1
        for poly in masks:  # list of (N,2)
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            x1, y1 = np.clip(pts.min(axis=0), [0, 0], [W-1, H-1])
            x2, y2 = np.clip(pts.max(axis=0), [0, 0], [W-1, H-1])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = [int(x1), int(y1), int(x2), int(y2)]
        return best

    crop_box = None
    # Prefer masks if available
    if getattr(r0, "masks", None) is not None and getattr(r0.masks, "xy", None):
        crop_box = bbox_from_masks(r0.masks.xy)

    # Fallback to highest-confidence box if no masks
    if crop_box is None and getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
        confs = r0.boxes.conf.cpu().numpy().tolist()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = r0.boxes.xyxy[idx].cpu().numpy().tolist()
        crop_box = [max(0, int(x1)), max(0, int(y1)), min(W-1, int(x2)), min(H-1, int(y2))]

    # If segmentation failed, use the whole image
    if crop_box is None:
        crop_box = [0, 0, W-1, H-1]

    # Slight padding (2% of max dimension) to avoid tight crops
    pad = int(0.02 * max(H, W))
    x1 = max(0, crop_box[0] - pad)
    y1 = max(0, crop_box[1] - pad)
    x2 = min(W, crop_box[2] + pad)
    y2 = min(H, crop_box[3] + pad)

    crop = src[y1:y2, x1:x2].copy()
    crop_path = Path(out_dir) / f"{img_path.stem}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    # ---- In-memory DocScanner dewarp (fallback to crop) ----
    rect_img = docscanner_dewarp_numpy(crop)
    if rect_img is None:
        rect_img = crop
    rect_img = stretch_to_ratio(rect_img,  ratio_w=420, ratio_h=297)
    # Save rectified/cropped image to SEG_OUT_DIR (single write)
    rect_path = Path(out_dir) / f"{img_path.stem}_rect.jpg"
    cv2.imwrite(str(rect_path), rect_img)

    # Create 12 ratio tiles from the selected image
    colcrop.crop_and_save(rect_path, Path(TILES_OUT_DIR), colcrop.CONFIG, flat=True)

    # Gather tiles for downstream detection
    tiles = []
    stem = rect_path.stem
    for r in [1, 2]:
        for c in range(1, 7):
            tpath = Path(TILES_OUT_DIR) / f"{stem}_R{r}C{c}.jpg"
            if tpath.exists():
                img = cv2.imread(str(tpath))
                if img is not None:
                    tiles.append((str(tpath), img))
            else:
                print(f"[WARN] missing tile: {tpath}")

    return tiles, rect_path

# ---------------- main per-image process ----------------
def process_image_pipeline(image_path: str, save_root: str = None):
    """
    Runs preprocess, then detection on each tile.
    If save_root is provided, per-tile inference artifacts are saved under:
      save_root/tiles/<tile_stem>_det.jpg|.txt|.json
    Returns: output, rect_path
    """
    img_list, rect_path = preprocess(image_path)
    output = data(0, 0, 0, 0, 0, 0, 0, [])

    tiles_save_dir = None
    if save_root is not None:
        tiles_save_dir = str(Path(save_root) / "tiles")
        Path(tiles_save_dir).mkdir(parents=True, exist_ok=True)

    for i, (filename, tile_img) in enumerate(img_list):
        output = get_selected_choices(i, tile_img, output, filename, save_dir=tiles_save_dir)

    return output, rect_path

# ---------------- CSV export ----------------
def export_data_to_csv(sbj_ls, is_valid, reasons):
    """
    sbj_ls: list of [color, (code, name)]
    Each code column will contain a numeric color id per your mapping:
      Pink 3, Blue 4, Orange 5, Brown 6
    sem3..sem6 hold comma joined codes by color group
    """
    color_map = {"Blue": 3, "Yellow": 4, "Red": 5, "Brown": 6}
    row = {c: '' for c in CSV_COLUMNS}

    for code in SUBJECT_CODES:
        row[code] = 0

    sem_buckets = {3: [], 4: [], 5: [], 6: []}

    for color_name, (code, _name) in sbj_ls:
        val = color_map.get(color_name)
        if val is None:
            continue
        if code in SUBJECT_CODES:
            row[code] = val
        sem_buckets[val].append(code)

    if sem_buckets[3]:
        row['sem3'] = ','.join(sem_buckets[3])
    if sem_buckets[4]:
        row['sem4'] = ','.join(sem_buckets[4])
    if sem_buckets[5]:
        row['sem5'] = ','.join(sem_buckets[5])
    if sem_buckets[6]:
        row['sem6'] = ','.join(sem_buckets[6])

    row['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    row['is_valid'] = bool(is_valid)
    row['reasons'] = '' if not reasons else ', '.join(reasons)
    row['hostname'] = hostname
    row['system_name'] = system_name
    row['processor'] = processor
    row['version'] = version
    row['machine'] = machine

    save_csv_append(row)

# ---------------- Mascot logic (dynamic) ----------------
# ---------------- Mascot logic (LLM designs hybrid name from full elective list) ----------------
import os, re, json, random, requests

EMOJI_CANDIDATES = [
    ("ฟิสิกส์|ควอนตัม|สนาม|กลศาสตร์|ดาราศาสตร์|แสง|ออปติก", "🌀"),
    ("คณิต|แคลคูลัส|สถิติ|พีชคณิต|กราฟ|เชิงจริง|อนุพันธ์", "🧮"),
    ("เคมี|สเปกโทร|จลนศาสตร์|สาร|โมเลกุล|วิเคราะห์", "⚗️"),
    ("ชีว|พันธุ|ประสาท|นิเวศ|ชีววิทยา|ชีวเวช", "🧬"),
    ("คอมพิวเตอร์|เขียนโปรแกรม|อัลกอริทึม|python|หุ่นยนต์|แมชชีน|ข้อมูล", "💻"),
    ("บูรณาการ|เชิงคำนวณ|ข้ามสาขา|นวัตกรรม|สหสาขา|วิศวกรรม", "🧩"),
    ("ศิลป์|ออกแบบ|ดนตรี|จิตรกรรม|ศิลปะ", "🎨"),
    ("ภาษา|วรรณกรรม|การสื่อสาร|ภาษาญี่ปุ่น|เกาหลี|จีน|รัสเซีย|สเปน|ฝรั่งเศส|เยอรมัน|อาหรับ|ไทย|อังกฤษ", "🗣️"),
]

LANG_PREFIXES = ("ท302","อ302","ก302","จ302","ซ302","ญ302","ป302","ฝ302","ย302","ร302")  # TH, EN, KR, CN, RU, JP, ES, FR, DE, AR

def count_language_subjects(subject_list):
    """
    subject_list: [[color, [code, name]], ...]
    returns (lang_count, lang_examples)
    """
    cnt, examples = 0, []
    for _, (code, name) in subject_list:
        if code.startswith(LANG_PREFIXES) or re.search(r"(ภาษา|วรรณกรรม|การสื่อสาร)", name):
            cnt += 1
            if len(examples) < 3:  # keep a few for prompting
                examples.append(f"{code}:{name}")
    return cnt, examples

def pick_emoji_from_text(text: str) -> str:
    for patt, e in EMOJI_CANDIDATES:
        if re.search(patt, text):
            return e
    return "🧩"

# ---- very small local fallback if LLM fails ----
ADJ_POOL = ["สุดปัง","สายโหด","สุดเท่","ขั้นเทพ","สายลุย","สุดคูล","สุดเฟี้ยว","หัวไว","สายไอเดีย","สายทดลอง","ข้ามศาสตร์"]
KEY_TERMS = ["ฟิสิกส์","ชีวเวช","ชีว","คอมพิวเตอร์","คณิตศาสตร์","เคมี","วิศวกรรม","ดาราศาสตร์","ศิลป์","ข้อมูล","หุ่นยนต์"]

def _fallback_name_from_subjects(subject_list):
    names = [n for _, (_, n) in subject_list]
    joined = " ".join(names)
    chosen = []
    for k in KEY_TERMS:
        if k in joined and k not in chosen:
            chosen.append(k)
        if len(chosen) == 2:
            break
    if not chosen:
        chosen = ["บูรณาการ"]
    adj = random.choice(ADJ_POOL)
    return f"ห่านนัก{''.join(chosen)}{adj}"

def _fallback_description(subject_list, lang_count):
    # short, friendly, data-aware; respects the language rule
    ex = [name for _, (_, name) in subject_list][:2]
    bits = []
    if ex:
        bits.append(f"(เช่น {(' และ ').join(ex)})")
    lang_note = "" if lang_count >= 3 else " วิชาภาษา 2 วิชาเป็นรายวิชาบังคับจึงไม่ได้นับเป็นความสนใจหลัก "
    return (
        f"คุณเป็นสายผสมที่หยิบวิชาที่ใช่แล้วจัดเต็ม{lang_note}"
        f"{''.join(bits)} หวังว่าเราจะได้เจอกันอีกที่นี่นะ!"
    )

# --------- Prompt: let the model invent a hybrid mascot name from full list ---------
SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยสร้าง 'มาสคอตห่าน' แบบคัสตอมจากรายการวิชาเลือกที่ผู้เรียนลงทะเบียนจริง\n"
    "กติกา:\n"
    "1) สร้างชื่อมาสคอตไทยเองในรูปแบบ: ห่านนัก<สาขาหรือสหสาขา><คำวิเศษณ์เท่ๆ>\n"
    "   - อนุญาตให้ผสมสาขา/คำ เช่น 'วิศวกรรมชีวเวช', 'ฟิสิกส์ดาราศาสตร์เชิงคำนวณ', 'คณิตศาสตร์ข้อมูล'\n"
    "   - ห้ามใช้ลิสต์ชื่อสำเร็จรูป ต้องคิดคำวิเศษณ์ให้เข้ากับวิชาที่ส่งมาจริง\n"
    "2) เขียนคำบรรยาย 4–5 ประโยค (ราว 120–220 ตัวอักษร) โทนสนุก เป็นกันเอง อ้างอิงชื่อวิชาจริง 1–2 วิชา\n"
    "3) กฎเรื่องวิชาภาษา: วิชาภาษา 2 วิชาเป็นข้อบังคับพื้นฐาน ไม่ควรสรุปว่าเป็นสายภาษาถ้ายอดรวมวิชาภาษาน้อยกว่า 3\n"
    "4) หลีกเลี่ยงการกล่าวอ้างความเชี่ยวชาญ ให้สะท้อน 'ความสนใจ' จากการเลือกวิชา\n"
    "รูปแบบผลลัพธ์ (JSON เท่านั้น):\n"
    "{'mascot':'ห่านนัก...<adj>', 'description':'<string>', 'signals':{'lang_count':<int>, 'focus_terms':[...]} }"
)

def call_gemma_dynamic(subject_list, return_debug=False):
    """
    Sends the full selected elective list to the LLM and lets it design the mascot.
    Enforces: language=2 is baseline (only highlight language if >=3).
    Returns: dict with mascot, description, emoji, and debug fields.
    """
    lang_count, lang_examples = count_language_subjects(subject_list)

    # If no API key/URL -> local fallback
    if not (GEMMA_API_URL and GEMMA_API_KEY):
        mascot = _fallback_name_from_subjects(subject_list)
        desc   = _fallback_description(subject_list, lang_count)
        text   = mascot + " " + desc
        return {
            "mascot": mascot,
            "description": desc,
            "emoji": pick_emoji_from_text(text),
            "llm_debug": {"used_fallback": True, "lang_count": lang_count, "lang_examples": lang_examples} if LLM_DEBUG else None
        }

    try:
        url = GEMMA_API_URL + ("&" if "?" in GEMMA_API_URL else "?") + f"key={GEMMA_API_KEY}"
        headers = {"Content-Type": "application/json"}

        # build a compact subject manifest for the model
        subj_manifest = [{"code": code, "name": name, "color": color} for (color, (code, name)) in subject_list]
        user_prompt = (
            SYSTEM_PROMPT + "\n\n"
            + "subjects_selected = " + json.dumps(subj_manifest, ensure_ascii=False) + "\n"
            + f"language_rule = '2 ภาษาเป็นข้อบังคับ; ให้ตีความเป็นความสนใจแท้จริงเมื่อ lang_count >= 3' \n"
            + f"lang_count = {lang_count}; lang_examples = " + json.dumps(lang_examples, ensure_ascii=False) + "\n"
            + "กรุณาส่งคืน JSON ตามรูปแบบกำหนดเท่านั้น"
        )

        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 600,
                "responseMimeType": "application/json"
            }
        }

        r = requests.post(url, headers=headers, json=payload, timeout=25)
        status = r.status_code
        body_head = r.text[:800]
        print(f"[LLM] status={status} url={GEMMA_API_URL}")
        print(f"[LLM] body_head={body_head}")
        r.raise_for_status()

        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        ).strip()

        obj = json.loads(text) if text else {}
        mascot = (obj.get("mascot") or "").strip()
        desc   = (obj.get("description") or "").strip()

        # Hard guards
        if not mascot.startswith("ห่านนัก"):
            # try to salvage; otherwise fallback synth
            mascot = mascot if mascot else _fallback_name_from_subjects(subject_list)
            if not mascot.startswith("ห่านนัก"):
                mascot = "ห่านนัก" + mascot

        if not desc:
            desc = _fallback_description(subject_list, lang_count)
        if not desc.endswith("หวังว่าเราจะได้เจอกันอีกที่นี่นะ!"):
            desc = desc.rstrip() + " หวังว่าเราจะได้เจอกันอีกที่นี่นะ!"

        # Emoji guess from both name & description
        emoji = pick_emoji_from_text(mascot + " " + desc)

        out = {
            "mascot": mascot,
            "description": desc,
            "emoji": emoji
        }
        if return_debug or LLM_DEBUG:
            out["llm_debug"] = {
                "status": status,
                "raw_candidates_head": body_head,
                "parsed_text_head": text[:300],
                "lang_count": lang_count
            }
        return out

    except Exception as e:
        print(f"[LLM][ERROR] {e}")
        mascot = _fallback_name_from_subjects(subject_list)
        desc   = _fallback_description(subject_list, lang_count)
        return {
            "mascot": mascot,
            "description": desc,
            "emoji": pick_emoji_from_text(mascot + " " + desc),
            "llm_debug": {"error": str(e), "lang_count": lang_count} if LLM_DEBUG else None
        }


# ---------------- Flask routes ----------------
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({'message': 'Test route reached'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_filename = timestamp + Path(file.filename).suffix
    filepath = Path(UPLOAD_FOLDER) / unique_filename
    file.save(str(filepath))

    # create a run folder under processed/<upload_stem>
    run_dir = Path(PROCESSED_FOLDER) / Path(unique_filename).stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # run pipeline and save per-tile inference outputs
    output, rect_path = process_image_pipeline(str(filepath), save_root=str(run_dir))
    is_valid, reasons = validate_output(output)

    # save annotated image (optional preview asset)
    annotated_name = f"{Path(filepath).stem}_annotated.jpg"
    save_annotated_image(str(rect_path), output.subject_list, annotated_name)

    # IMPORTANT: do NOT save CSV here. We only return the result.
    response = {
        'validity': is_valid,
        'reasons': reasons if not is_valid else None,
        'subject_list': output.subject_list
    }
    print(response)
    return jsonify(response), 200

@app.route('/confirm', methods=['POST'])
def confirm_save():
    """
    Body: {
      "subject_list": [[color, [code, name]], ...],
      "validity": true/false,
      "reasons": [ ... ]  # optional
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        subject_list = payload.get("subject_list", [])
        validity = bool(payload.get("validity", False))
        reasons = payload.get("reasons") or []

        # Write CSV now (only on confirm)
        export_data_to_csv(subject_list, validity, reasons)
        return jsonify({"saved": True}), 200
    except Exception as e:
        return jsonify({"saved": False, "error": str(e)}), 500



# NEW: per-record mascot + dynamic description (LLM or fallback)
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Body: {"subject_list": [[color, [code, name]], ...]}
    Returns:
      {
        "mascot": "<Thai name>",          # e.g., ห่านนักวิศวกรรมชีวเวชสุดปัง
        "description": "<Thai text>",     # ends with “หวังว่าเราจะได้เจอกันอีกที่นี่นะ!”
        "emoji": "🧬"                      # guessed from content
      }
    """
    try:
        payload = request.get_json(force=True) or {}
        subject_list = payload.get("subject_list", [])
        out = call_gemma_dynamic(subject_list)
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # tweak host or port as needed
    app.run(host="0.0.0.0", port=5000, debug=True)
