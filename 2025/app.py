from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, base64, os, json, re
from pathlib import Path
import uuid
from datetime import datetime
import requests

# --- your doc scanning pipeline ---
from test import process_image  # keep your import

app = Flask(__name__)

# Folders
SAVE_DIR = Path("captured"); SAVE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = SAVE_DIR / "results"; RESULTS_DIR.mkdir(exist_ok=True)

# YOLO model (loaded in __main__)
your_model_instance = None  # will be assigned in __main__

# -----------------------------
# Subject bucketing
# -----------------------------
def categorize_subject(name: str, code: str):
    """
    Map real subject names/codes → bucket:
    ['MATH','PHYS','CHEM','BIO','CS','INT','ART','LANG']
    """
    t = f"{code} {name}".lower()

    KW = [
        (r"(คณิต|calculus|สถิติ|พีชคณิต|ค302|graph|stochastic)", "MATH"),
        (r"(ฟิสิกส์|กลศาสตร์|แรง|สนาม|ควอนตัม|ว303|ฟิสิกส์เชิงคำนวณ)", "PHYS"),
        (r"(เคมี|วัสดุ|จลนศาสตร์|สเปกโทร|ว304|chem)", "CHEM"),
        (r"(ชีว|พันธุ|ประสาท|จุลชีว|นิเวศ|ว305)", "BIO"),
        (r"(เขียนโปรแกรม|python|อัลกอริทึม|วิทยาการคอมพิวเตอร์|หุ่นยนต์|machine learning|ว307|คอมพิวเตอร์|ชีวสารสนเทศ)", "CS"),
        (r"(บูรณาการ|integrat|วิทยาศาสตร์เชิงคำนวณ|โครงงาน|ว3080[78])", "INT"),
        (r"(ศิลป์|ออกแบบ|จิตรกรรม|ดนตรี|ศ302|art|design)", "ART"),
        (r"(ภาษา|วรรณกรรม|สื่อสาร|ญ302|ก302|จ302|ป302|ฝ302|ย302|ร302|ซ302|อ302|ท302|ส302|english|japanese|chinese)", "LANG"),
    ]
    for pat, lab in KW:
        if re.search(pat, t):
            return lab

    # Code-based fallbacks (typical Thai HS/KVIS style)
    if code.startswith("ค3"): return "MATH"
    if code.startswith("ว303"): return "PHYS"
    if code.startswith("ว304"): return "CHEM"
    if code.startswith("ว305"): return "BIO"
    if code.startswith("ว307"): return "CS"
    if code.startswith("ว308"): return "INT"
    if code.startswith("ศ3"): return "ART"
    if code and code[0] in "กจซญปฝยายรรทอส": return "LANG"
    return "INT"

MASCOTS = {
    "PHYS": "ห่านนักฟิสิกส์สายโหด",
    "MATH": "ห่านนักคณิตศาสตร์สุดเท่",
    "CHEM": "ห่านนักเคมีขั้นเทพ",
    "BIO" : "ห่านนักชีวะสุดปัง",
    "CS"  : "ห่านโปรแกรมเมอร์เจ๋งเป้ง",
    "INT" : "ห่านบูรณาการสุดคูล",
    "ART" : "ห่านศิลป์สุดเฟี้ยว",
    "LANG": "ห่านภาษาสารพัดนึก",
}
MASCOT_EMOJI = {
    "PHYS":"🌀", "MATH":"🧮", "CHEM":"⚗️", "BIO":"🧬",
    "CS":"💻", "INT":"🧩", "ART":"🎨", "LANG":"🗣️",
}
TH_LABELS = {
    "PHYS":"นักฟิสิกส์", "MATH":"นักคณิต", "CHEM":"นักเคมี", "BIO":"สายชีวะ",
    "CS":"โปรแกรมเมอร์", "INT":"สายบูรณาการ", "ART":"สายศิลป์", "LANG":"สายภาษา"
}

def summarize_subjects(subject_list):
    """
    subject_list: [[color, [code, name]], ...]
    returns (counts, details, primary_bucket)
    """
    counts = {k:0 for k in MASCOTS}
    details = {k:[] for k in MASCOTS}
    for color, pair in subject_list:
        code, name = pair
        lab = categorize_subject(name, code)
        counts[lab] += 1
        details[lab].append({"code": code, "name": name, "color": color})

    # pick primary by max count; if mixes many buckets, prefer INT
    winner = max(counts.items(), key=lambda kv: kv[1])[0]
    distinct = sum(1 for v in counts.values() if v>0)
    if distinct >= 4 and counts[winner] <= 2:
        winner = "INT"
    return counts, details, winner

def compose_desc_thai(counts, details):
    """
    Dynamic Thai description built from actual buckets/subjects.
    Always ends with the closing sentence.
    """
    top2 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    majors = [k for k,v in top2 if v>0]

    subj_names = []
    for k in majors:
        subj_names += [it["name"] for it in details.get(k, [])][:2]
    subj_names = subj_names[:3]

    if len(majors) == 0:
        core = "ชอบหลายแนวและกำลังค้นหาตัวเอง"
    elif len(majors) == 1:
        core = f"เป็น{TH_LABELS[majors[0]]}ตัวจริง"
    else:
        core = f"เป็นทั้ง{TH_LABELS[majors[0]]}และ{TH_LABELS[majors[1]]} ผสมผสานได้ลงตัว"

    tail = f" โดยเฉพาะ {', '.join(subj_names)}" if subj_names else ""
    return f"คุณ{core}{tail} หวังว่าเราจะได้เจอกันอีกที่นี่นะ!"

# -----------------------------
# LLM (Gemma 3 27B) + strict JSON
# -----------------------------
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "")  # e.g. your provider endpoint
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "")
GEMMA_MODEL   = os.getenv("GEMMA_MODEL", "gemma-3-27b-instruct")

SYSTEM_RULES = """คุณคือระบบคัดเลือกมาสคอตให้เด็กจากรายวิชาที่เด็กเลือกจริง
คุณต้อง:
1) เลือกมาสคอตที่ ‘เข้ากับความชอบของเด็กที่สุด’ เพียง 1 ตัว จากรายการต่อไปนี้ (และให้โค้ดหมวด):
   - ห่านนักฟิสิกส์สายโหด (PHYS)
   - ห่านนักคณิตศาสตร์สุดเท่ (MATH)
   - ห่านนักเคมีขั้นเทพ (CHEM)
   - ห่านนักชีวะสุดปัง (BIO)
   - ห่านโปรแกรมเมอร์เจ๋งเป้ง (CS)
   - ห่านบูรณาการสุดคูล (INT)
   - ห่านศิลป์สุดเฟี้ยว (ART)
   - ห่านภาษาสารพัดนึก (LANG)
2) เขียน “description” เป็นภาษาไทยสั้นๆ น่ารักๆ (≤ 200 ตัวอักษร) อ้างอิงจากรายวิชาที่เด็กเลือก ‘ชุดนี้จริงๆ’
   - สรุปภาพรวมความชอบ (เช่นผสมฟิสิกส์+ศิลป์เป็นต้น)
   - ใส่ชื่อวิชาที่เกี่ยวข้องบางวิชาเพื่อให้ดูเป็นส่วนตัว
   - จบด้วยประโยค: “หวังว่าเราจะได้เจอกันอีกที่นี่นะ!”
3) ให้เหตุผลสั้นๆ 2–4 ข้อ ว่าทำไมมาสคอตนี้เหมาะ

รูปแบบตอบกลับต้องเป็น JSON เท่านั้น:
{
  "mascot": "...ชื่อมาสคอต...",
  "code": "PHYS|MATH|CHEM|BIO|CS|INT|ART|LANG",
  "description": "...ข้อความ...",
  "reasons": ["...", "..."]
}
ห้ามใช้ข้อความสำเร็จรูป ห้ามเดาความชอบที่ไม่มีในข้อมูล ห้ามส่งสิ่งอื่นนอกจาก JSON"""

def call_llm_gemma(subjects_summary, primary_guess, counts, details):
    """
    subjects_summary: [{'code':'ค30205','name':'...'}, ...]
    primary_guess: bucket code (or None)
    """
    # Offline / unset → compose dynamically
    if not (GEMMA_API_URL and GEMMA_API_KEY):
        code = primary_guess or "INT"
        return {
            "mascot": MASCOTS[code],
            "code": code,
            "description": compose_desc_thai(counts, details),
            "reasons": ["อิงจากรายวิชาที่เลือกจริง", "ส่วนผสมของหมวดวิชานำไปสู่มาสคอตนี้"],
        }

    try:
        headers = {"Authorization": f"Bearer {GEMMA_API_KEY}", "Content-Type":"application/json"}
        payload = {
            "model": GEMMA_MODEL,
            "input": f"{SYSTEM_RULES}\n\nSubjects:\n{json.dumps(subjects_summary, ensure_ascii=False)}\nPrimary bucket: {primary_guess}",
            "response_format": {"type":"json_object"},
            "temperature": 0.5,
            "max_output_tokens": 400,
        }
        r = requests.post(GEMMA_API_URL, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()

        txt = (
            data.get("content")
            or (data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text"))
            or (data.get("choices",[{}])[0].get("message",{}).get("content"))
        )
        parsed = json.loads(txt)

        if all(k in parsed for k in ("mascot","code","description")):
            if "หวังว่าเราจะได้เจอกันอีกที่นี่นะ!" not in parsed["description"]:
                parsed["description"] = parsed["description"].rstrip() + " หวังว่าเราจะได้เจอกันอีกที่นี่นะ!"
            return parsed
    except Exception as e:
        print("LLM error / fallback:", e)

    code = primary_guess or "INT"
    return {
        "mascot": MASCOTS[code],
        "code": code,
        "description": compose_desc_thai(counts, details),
        "reasons": ["อิงจากรายวิชาที่เลือกจริง", "ส่วนผสมของหมวดวิชานำไปสู่มาสคอตนี้"],
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index2.html")

# Base64 JSON route (kept for compatibility)
@app.route("/capture", methods=["POST"])
def capture():
    try:
        data = request.json["image"]
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        file_id = str(uuid.uuid4())
        img_path = SAVE_DIR / f"{file_id}.jpg"
        cv2.imwrite(str(img_path), img)

        process_image(
            model=your_model_instance,
            img_path=img_path,
            out_dir=RESULTS_DIR,
            imgsz=640, conf=0.25, iou=0.5,
            extend_iters=6, do_warp=True
        )

        crop_path = RESULTS_DIR / f"{img_path.stem}_rect.jpg"
        if not crop_path.exists():
            return jsonify({"error": "Cropping failed"})

        with open(crop_path, "rb") as f:
            crop_base64 = base64.b64encode(f.read()).decode("utf-8")

        # NOTE: fill subject_list/validity/reasons from your pipeline if available
        return jsonify({
            "image": f"data:image/png;base64,{crop_base64}",
            "validity": True,
            "reasons": [],
            "subject_list": []  # [[color,[code,name]], ...]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Multipart upload route (matches your JS)
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error":"no file"}), 400

        file_id = str(uuid.uuid4()); img_path = SAVE_DIR / f"{file_id}.jpg"
        file.save(str(img_path))

        process_image(
            model=your_model_instance,
            img_path=img_path,
            out_dir=RESULTS_DIR,
            imgsz=640, conf=0.25, iou=0.5,
            extend_iters=6, do_warp=True
        )
        crop_path = RESULTS_DIR / f"{img_path.stem}_rect.jpg"
        if not crop_path.exists():
            return jsonify({"error": "Cropping failed"})

        with open(crop_path, "rb") as f:
            crop_base64 = base64.b64encode(f.read()).decode("utf-8")

        # TODO: replace stubs with actual outputs from your scanner
        detected_subjects = []  # e.g. [['Blue',['ว30301','กลศาสตร์คลาสสิก']], ...]
        validity = True
        reasons = []

        return jsonify({
            "image": f"data:image/png;base64,{crop_base64}",
            "validity": validity,
            "reasons": reasons,
            "subject_list": detected_subjects
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# LLM mascot recommendation (fresh per record)
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Input JSON: { "subject_list": [[color,[code,name]], ...] }
    Output JSON: { winner, mascot, emoji, description, reasons, counts, details }
    """
    try:
        data = request.get_json(force=True)
        subject_list = data.get("subject_list", [])

        counts, details, primary = summarize_subjects(subject_list)

        flat = []
        for bucket, arr in details.items():
            for it in arr:
                flat.append({"code": it["code"], "name": it["name"]})

        out = call_llm_gemma(
            subjects_summary=flat,
            primary_guess=primary,
            counts=counts,
            details=details
        )
        out["emoji"] = MASCOT_EMOJI.get(out.get("code","INT"), "🧩")

        return jsonify({
            "winner": out["code"],
            "mascot": out["mascot"],
            "emoji": out["emoji"],
            "description": out["description"],
            "reasons": out.get("reasons", []),
            "counts": counts,
            "details": details
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    from ultralytics import YOLO
    # Load once
    your_model_instance = YOLO("runs_paper_seg/y11n_baseline4/weights/best.pt")
    # Optionally set:
    # export GEMMA_API_URL="https://<your-gemma-endpoint>"
    # export GEMMA_API_KEY="<your-key>"
    # export GEMMA_MODEL="gemma-3-27b-instruct"
    app.run(host="0.0.0.0", port=5000, debug=True)
