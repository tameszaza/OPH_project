# api.py
# Flask API for subject enrollment scanning
# Pipeline:
#   1) YOLO segmentation isolates the document and warps to a flat rectangle
#   2) Ratio-based grid cropping splits the rectified doc into 12 tiles (R1C1..R2C6)
#   3) Per-tile detection with your best6.pt model to pair sticker colors with nearest codes
#   4) Validate and append a local CSV record

from flask import Flask, request, jsonify, render_template
import os
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import socket
import platform

# ===== import your two helper modules =====
# Make sure these two files sit next to this api.py, or are importable on PYTHONPATH
import test as docseg              # your segmentation and rectification utilities
import tt as colcrop      # your 12-tile ratio cropper

app = Flask(__name__)

# -------------------- constants and folders --------------------
UPLOAD_FOLDER = 'uploads'
SEG_OUT_DIR = 'seg_out'
TILES_OUT_DIR = 'tiles_out'
PROCESSED_FOLDER = 'processed'   # optional annotated dumps
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
# Segmentation weights:
#   If DOC_SEG_WEIGHTS env var is set, it is used. Otherwise we take the latest runs_paper_seg/*/weights/best.pt
DOC_SEG_WEIGHTS = 'runs_paper_seg/y11n_baseline4/weights/best.pt'
if DOC_SEG_WEIGHTS and Path(DOC_SEG_WEIGHTS).exists():
    seg_model = YOLO(DOC_SEG_WEIGHTS)
else:
    seg_model = YOLO(docseg.latest_best())

# Detection weights for sticker+code boxes
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
    # make sure all columns exist
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = ''
    return df[CSV_COLUMNS].copy()

def save_csv_append(row_dict: dict):
    df = load_csv()
    # count
    try:
        last_count = int(df['count'].replace('', '0').astype(int).max()) if len(df) else 0
    except Exception:
        last_count = 0
    row_dict['count'] = last_count + 1
    # row as one-row DataFrame with all columns
    out = {c: '' for c in CSV_COLUMNS}
    out.update({k: v for k, v in row_dict.items() if k in CSV_COLUMNS})
    df = pd.concat([df, pd.DataFrame([out])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

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
    """
    image_path: path to the original (rectified) image
    detections: list of [color_name, (code, name)]
    output_name: filename to save in PROCESSED_FOLDER
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read {image_path} for annotation")
        return

    for color_name, (code, name) in detections:
        # Draw text at the top-left with the subject code and color
        cv2.putText(
            img,
            f"{color_name} - {code}",
            (10, 30 + 30 * detections.index([color_name, (code, name)])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    save_path = Path(PROCESSED_FOLDER) / output_name
    cv2.imwrite(str(save_path), img)
    print(f"[INFO] Annotated image saved to {save_path}")

# ---------------- tiny utils ----------------
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea + 1e-9)

def get_selected_choices(i, img, output: data, filename_for_log: str, save_dir: str = None):
    """
    Runs detection on a single tile, updates output, and optionally saves
    per-tile inference artifacts into save_dir:
      - <tile_stem>_det.jpg  annotated image
      - <tile_stem>_det.txt  human readable lines
      - <tile_stem>_det.json JSON list of dicts
    """
    model = yolo_det_model
    results = model(img, verbose=False)
    dets = results[0].boxes

    # names map
    if hasattr(results[0], "names") and isinstance(results[0].names, dict):
        id2name = results[0].names
    else:
        id2name = {0: "Blue", 1: "Yellow", 2: "Red", 3: "Brown", 4: "Code"}

    # ---------- optional save of inference outputs ----------
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

    # NMS by IoU across stickers of any color
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

    # Pre-sort codes by y for vertical order mapping
    sorted_codes = sorted(code_boxes, key=lambda x: x[0][1])  # by top y

    # helper to compute centers
    def _center(box):
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    # For each sticker, find nearest code, then drop if sticker is on the right side of that code
    for sticker_bbox, _, color in sticker_boxes:
        if not code_boxes:
            continue

        sx, sy = _center(sticker_bbox)

        # find nearest code by Euclidean center distance
        nearest_box = None
        min_d = float('inf')
        for code_bbox, _ in code_boxes:
            cx, cy = _center(code_bbox)
            d = np.hypot(sx - cx, sy - cy)
            if d < min_d:
                min_d = d
                nearest_box = code_bbox

        # safety tolerance relative to code width to avoid jitter cuts
        cx, cy = _center(nearest_box)
        code_w = max(1.0, nearest_box[2] - nearest_box[0])
        x_tolerance = 0.05 * code_w  # 5 percent of code width

        # DROP RULE: ignore sticker if it lies to the right of its closest code (center-to-center)
        # i.e., sticker center x >= code center x - tiny tolerance
        if sx >= cx - x_tolerance:
            # skip this sticker
            continue

        # if kept, assign order by the index of nearest_box within sorted_codes
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


# ---------------- segmentation + tiling preprocess ----------------
def preprocess(image_path: str):
    img_path = Path(image_path)
    out_dir = Path(SEG_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    docseg.process_image(
        seg_model, img_path, out_dir,
        imgsz=960, conf=0.25, iou=0.5,
        extend_iters=6, do_warp=True, rect_only=True
    )
    rect_path = out_dir / f"{img_path.stem}_rect.jpg"
    if not rect_path.exists():
        rect_path = img_path

    colcrop.crop_and_save(rect_path, Path(TILES_OUT_DIR), colcrop.CONFIG, flat=True)

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
    img_list, rect_path = preprocess(image_path)  # unpack both values
    output = data(0, 0, 0, 0, 0, 0, 0, [])

    # prepare save directory for tiles
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

    # subject code columns default to 0 if you prefer numeric
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

    # fill sem lists
    if sem_buckets[3]:
        row['sem3'] = ','.join(sem_buckets[3])
    if sem_buckets[4]:
        row['sem4'] = ','.join(sem_buckets[4])
    if sem_buckets[5]:
        row['sem5'] = ','.join(sem_buckets[5])
    if sem_buckets[6]:
        row['sem6'] = ','.join(sem_buckets[6])

    # meta
    row['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    row['is_valid'] = bool(is_valid)
    row['reasons'] = '' if not reasons else ', '.join(reasons)
    row['hostname'] = hostname
    row['system_name'] = system_name
    row['processor'] = processor
    row['version'] = version
    row['machine'] = machine

    save_csv_append(row)

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

    # save annotated image
    annotated_name = f"{Path(filepath).stem}_annotated.jpg"
    save_annotated_image(str(rect_path), output.subject_list, annotated_name)

    response = {
        'validity': is_valid,
        'reasons': reasons if not is_valid else None,
        'subject_list': output.subject_list
    }

    # write CSV locally
    export_data_to_csv(output.subject_list, is_valid, reasons)
    print(response)
    return jsonify(response), 200

if __name__ == '__main__':
    # tweak host or port as needed
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "cert.key"))
