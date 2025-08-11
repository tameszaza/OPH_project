from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from pathlib import Path
import uuid

# Import your doc scanning pipeline
from test import process_image  # Replace with your actual import

app = Flask(__name__)

# Folder to save captured & cropped images
SAVE_DIR = Path("captured")
SAVE_DIR.mkdir(exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    try:
        data = request.json["image"]
        # Strip header: data:image/png;base64,xxxx
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save original capture
        file_id = str(uuid.uuid4())
        img_path = SAVE_DIR / f"{file_id}.jpg"
        cv2.imwrite(str(img_path), img)

        # Process image with your scanner
        results_dir = SAVE_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        process_image(
            model=your_model_instance,   # Pass your loaded YOLO model
            img_path=img_path,
            out_dir=results_dir,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            extend_iters=6,
            do_warp=True
        )

        # Path to cropped image
        crop_path = results_dir / f"{img_path.stem}_rect.jpg"
        if not crop_path.exists():
            return jsonify({"error": "Cropping failed"})

        # Read cropped image as base64
        with open(crop_path, "rb") as f:
            crop_base64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"image": f"data:image/png;base64,{crop_base64}"})

    except Exception as e:
        return jsonify({"error": str(e)})

# app.py
if __name__ == "__main__":
    from ultralytics import YOLO
    your_model_instance = YOLO("runs_paper_seg/y11n_baseline4/weights/best.pt")
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "cert.key"))



