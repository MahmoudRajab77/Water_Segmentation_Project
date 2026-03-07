"""
Flask deployment for Water Segmentation Model
Supports:
- 9-band TIFF satellite images
- RGB images
- Optional ground truth mask
- Metrics calculation (IoU, Precision, Recall, F1)
"""

import os
import io
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
import tifffile
from werkzeug.utils import secure_filename

from pyngrok import ngrok
import threading
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import PretrainedUNet


# =========================
# Flask setup
# =========================

app = Flask(__name__)
run_with_ngrok(app)

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

MODEL_URL = "https://github.com/MahmoudRajab77/Water_Segmentation_Project/releases/download/v1.0.0/best_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'tif','tiff','png','jpg','jpeg'}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load model
# =========================

print("Loading model...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model = PretrainedUNet(n_channels=9, n_classes=1).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

print("Model loaded successfully")


# =========================
# Helper Functions
# =========================

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_bytes, filename):

    filename = secure_filename(filename)

    temp_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(temp_path,'wb') as f:
        f.write(image_bytes)

    print("Saved image to:", temp_path)

    # Load image
    if filename.endswith((".tif",".tiff")):

        image = tifffile.imread(temp_path)

    else:

        image = np.array(Image.open(temp_path).convert("RGB"))

    h,w = image.shape[:2]

    # Ensure 9 bands
    if image.ndim == 2:

        image = np.stack([image]*9, axis=-1)

    if image.shape[2] < 9:

        new_img = np.zeros((h,w,9))
        new_img[:,:,:image.shape[2]] = image
        image = new_img

    image = image[:,:,:9]

    tensor = torch.from_numpy(image).float().permute(2,0,1)

    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(128,128),
        mode="bilinear",
        align_corners=False
    )

    return tensor.to(DEVICE), temp_path


def create_overlay(image_path, mask_tensor):

    img = Image.open(image_path).convert("RGB").resize((128,128))
    img = np.array(img)

    mask = mask_tensor.squeeze().cpu().numpy()
    mask = mask > 0.5

    overlay = img.copy()

    overlay[mask] = [0,0,255]

    blended = (0.6*img + 0.4*overlay).astype(np.uint8)

    return Image.fromarray(blended)


def encode_base64(pil_image):

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def calculate_metrics(pred_mask, true_mask):

    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    intersection = np.logical_and(pred,true).sum()
    union = np.logical_or(pred,true).sum()

    iou = intersection / union if union else 0

    tp = intersection
    fp = np.logical_and(pred,~true).sum()
    fn = np.logical_and(~pred,true).sum()

    precision = tp/(tp+fp) if (tp+fp) else 0
    recall = tp/(tp+fn) if (tp+fn) else 0

    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0

    return {
        "iou": round(iou,4),
        "precision": round(precision,4),
        "recall": round(recall,4),
        "f1": round(f1,4)
    }


# =========================
# Routes
# =========================

@app.route("/")
def home():

    return "<h2>Water Segmentation API is running</h2>"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({"error":"No image uploaded"})


        image_file = request.files["image"]

        if image_file.filename == "":

            return jsonify({"error":"No image selected"})


        if not allowed_file(image_file.filename):

            return jsonify({"error":"Invalid file type"})


        image_file.stream.seek(0)

        image_bytes = image_file.read()

        input_tensor, temp_path = prepare_image(
            image_bytes,
            image_file.filename
        )

        # ====================
        # Inference
        # ====================

        with torch.no_grad():

            output = model(input_tensor)

            probs = torch.sigmoid(output)

            pred_mask = (probs > 0.5).float()

        # ====================
        # Mask image
        # ====================

        mask_np = (pred_mask[0,0].cpu().numpy()*255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)

        # Overlay
        overlay = create_overlay(temp_path, pred_mask)

        mask_base64 = encode_base64(mask_pil)
        overlay_base64 = encode_base64(overlay)

        # ====================
        # Stats
        # ====================

        water_pixels = int(pred_mask.sum().item())
        total_pixels = int(pred_mask.numel())

        water_percentage = (water_pixels/total_pixels)*100

        avg_confidence = probs.mean().item()*100

        # Pixel area (Sentinel-2)
        pixel_area_km2 = (30*30)/1_000_000
        area_km2 = water_pixels * pixel_area_km2


        response = {

            "success": True,

            "water_percentage": round(water_percentage,2),

            "avg_confidence": round(avg_confidence,2),

            "area_km2": round(area_km2,3),

            "water_pixels": water_pixels,

            "total_pixels": total_pixels,

            "mask_image": f"data:image/png;base64,{mask_base64}",

            "overlay_image": f"data:image/png;base64,{overlay_base64}"
        }


        # ====================
        # Ground truth metrics
        # ====================

        if "mask" in request.files:

            mask_file = request.files["mask"]

            if mask_file.filename != "":

                gt = Image.open(mask_file).convert("L")

                gt = gt.resize((128,128))

                gt = np.array(gt)

                gt = (gt > 128).astype(np.uint8)

                pred_np = pred_mask[0,0].cpu().numpy()

                metrics = calculate_metrics(pred_np, gt)

                response.update(metrics)


        return jsonify(response)


    except Exception as e:

        print("ERROR:", e)

        return jsonify({"error": str(e)})



# Running Ngrok in separated Thread 
def start_ngrok():
    time.sleep(2)
    public_url = ngrok.connect(5000).public_url
    print(f"\nPublic URL: {public_url}")
    print("Open this URL in your browser!")

threading.Thread(target=start_ngrok, daemon=True).start()

# =========================
# Run server
# =========================

if __name__ == "__main__":

    app.run()
