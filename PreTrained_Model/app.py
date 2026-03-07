"""
Flask deployment script for water segmentation model.
Accepts 9-band multispectral images and returns segmentation masks.
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
from PIL import Image
import tifffile
from werkzeug.utils import secure_filename
# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import PretrainedUNet
import requests
from tqdm import tqdm
import os






# ========== GITHUB RELEASES SETUP ==========
MODEL_PATH = 'best_model.pth'
RELEASE_URL = 'https://github.com/MahmoudRajab77/Water_Segmentation_Project/releases/download/v1.0.0/best_model.pth'

# Download model from GitHub Releases if not exists
if not os.path.exists(MODEL_PATH):
    print(f"📥 Downloading model from GitHub Releases...")
    
    response = requests.get(RELEASE_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(MODEL_PATH, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=MODEL_PATH) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    print(f"✅ Model downloaded successfully!")
else:
    print(f"✅ Model found locally at {MODEL_PATH}")






#----------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)

# ========== CONFIGURATION ==========
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== LOAD MODEL ==========
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = PretrainedUNet(n_channels=9, n_classes=1).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded successfully! Best IoU: {checkpoint.get('train_iou', 0):.4f}")

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_bytes, original_filename):
    """
    Convert uploaded image to tensor in the format expected by model.
    """
    # Save uploaded file temporarily
    filename = secure_filename(original_filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(temp_path, 'wb') as f:
        f.write(image_bytes)
    
    # Load image
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        # Multi-band TIFF
        image = tifffile.imread(temp_path)
        
        # Check if we have enough bands
        if len(image.shape) == 3 and image.shape[2] >= 9:
            # Use first 9 bands
            image = image[:, :, :9]
        else:
            # Handle case with fewer bands
            h, w = image.shape[:2]
            if len(image.shape) == 2:
                image_9band = np.stack([image] * 9, axis=2)
            else:
                image_9band = np.zeros((h, w, 9), dtype=image.dtype)
                image_9band[:, :, :3] = image[:, :, :3]
                for i in range(3, 9):
                    image_9band[:, :, i] = image[:, :, i % 3]
            image = image_9band
    else:
        # RGB image (JPEG/PNG)
        pil_image = Image.open(temp_path).convert('RGB')
        image = np.array(pil_image)
        h, w = image.shape[:2]
        image_9band = np.zeros((h, w, 9), dtype=image.dtype)
        image_9band[:, :, :3] = image
        for i in range(3, 9):
            image_9band[:, :, i] = image[:, :, i % 3]
        image = image_9band
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    
    # Resize to 128x128 if needed
    if image_tensor.shape[1] != 128 or image_tensor.shape[2] != 128:
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), 
            size=(128, 128), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Clean up temp file
    os.remove(temp_path)
    
    return image_tensor

def mask_to_image(mask_tensor):
    """Convert mask tensor to PIL Image for response"""
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)

def encode_image_to_base64(pil_image):
    """Encode PIL image to base64 string for JSON response"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# ========== ROUTES ==========
@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌊 Water Segmentation AI</title>
    <style>
        * {
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            background: #e8eaff;
            border-color: #764ba2;
        }
        
        .upload-area i {
            font-size: 60px;
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .upload-area p {
            margin: 5px 0;
            color: #444;
        }
        
        .file-input {
            display: none;
        }
        
        .file-name {
            margin-top: 15px;
            color: #667eea;
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            margin-top: 40px;
        }
        
        .images-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .image-card {
            flex: 1;
            min-width: 300px;
            background: #f5f5f5;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .image-card h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .image-card img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        
        .classification-result {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(67, 233, 123, 0.3);
        }
        
        .result-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 30px;
            color: white;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .result-text {
            font-size: 48px;
            font-weight: bold;
            color: white;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .confidence-bar {
            max-width: 500px;
            margin: 20px auto;
            background: rgba(255,255,255,0.3);
            border-radius: 30px;
            height: 30px;
            overflow: hidden;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            background: white;
            border-radius: 30px;
            transition: width 1s ease-in-out;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            color: #333;
            font-weight: bold;
            font-size: 14px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stat-card .value {
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-card .label {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            border-left: 4px solid #c62828;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 20px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
        }
        
        .badge {
            background: #4CAF50;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 12px;
            display: inline-block;
            margin-right: 10px;
        }
        
        .pixel-info {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 Satellite Water Detection</h1>
        <p class="subtitle">AI-powered water segmentation using U-Net + EfficientNet (9 spectral bands)</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <i>📤</i>
            <p>Click to upload or drag & drop</p>
            <p style="font-size: 12px; color: #999;">Supports: .tif, .tiff, .png, .jpg, .jpeg</p>
            <input type="file" id="fileInput" class="file-input" accept=".tif,.tiff,.png,.jpg,.jpeg">
        </div>
        <div id="fileName" class="file-name"></div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Analyzing image... This may take a few seconds</p>
        </div>
        
        <div id="errorMessage" class="error-message" style="display: none;"></div>
        
        <div id="results" class="results">
            <div class="images-container">
                <div class="image-card">
                    <h3>📷 Original Image</h3>
                    <img id="originalImage" src="" alt="Original Image">
                </div>
                <div class="image-card">
                    <h3>🧪 Water Detection Mask</h3>
                    <img id="maskImage" src="" alt="Segmentation Mask">
                    <div class="pixel-info" id="pixelInfo"></div>
                </div>
            </div>
            
            <div class="classification-result">
                <span class="result-badge">🌊 Water Detection Result</span>
                <div class="result-text" id="classificationText">Analyzing...</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill" style="width: 0%">0%</div>
                </div>
                <p style="color: white; margin-top: 10px;" id="confidenceDetail">Confidence score based on pixel analysis</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Water Coverage</div>
                    <div class="value" id="waterPercentage">0%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Water Area</div>
                    <div class="value" id="areaValue">0 km²</div>
                </div>
                <div class="stat-card">
                    <div class="label">Water Pixels</div>
                    <div class="value" id="pixelCount">0</div>
                </div>
            </div>
            
            <div style="text-align: center;">
                <button class="btn" onclick="resetUpload()">🔄 Analyze New Image</button>
            </div>
        </div>
        
        <div class="footer">
            <span class="badge">🚀 v2.0</span>
            <span>Model: U-Net + EfficientNet-b4 | 9 spectral bands | IoU: 0.803</span>
        </div>
    </div>
    
    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const errorMessage = document.getElementById('errorMessage');
        const originalImage = document.getElementById('originalImage');
        const maskImage = document.getElementById('maskImage');
        const pixelInfo = document.getElementById('pixelInfo');
        const waterPercentage = document.getElementById('waterPercentage');
        const areaValue = document.getElementById('areaValue');
        const pixelCount = document.getElementById('pixelCount');
        const classificationText = document.getElementById('classificationText');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceDetail = document.getElementById('confidenceDetail');
        
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            fileName.textContent = `📁 ${file.name}`;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                originalImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            uploadAndAnalyze(file);
        });
        
        async function uploadAndAnalyze(file) {
            results.style.display = 'none';
            errorMessage.style.display = 'none';
            loading.style.display = 'block';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    maskImage.src = data.mask_image;
                    
                    waterPercentage.textContent = data.water_percentage + '%';
                    areaValue.textContent = data.area_km2 + ' km²';
                    pixelCount.textContent = data.water_pixels.toLocaleString();
                    pixelInfo.textContent = `Total pixels: ${data.total_pixels} | Water pixels: ${data.water_pixels}`;
                    
                    const confidence = data.water_percentage;
                    let resultText = '';
                    let resultColor = '';
                    
                    if (confidence >= 70) {
                        resultText = '🌊 High Water Concentration';
                        resultColor = '#43e97b';
                    } else if (confidence >= 30) {
                        resultText = '💧 Moderate Water Presence';
                        resultColor = '#f9d423';
                    } else {
                        resultText = '🏝️ Low Water / Mostly Land';
                        resultColor = '#fa709a';
                    }
                    
                    classificationText.textContent = resultText;
                    confidenceFill.style.width = confidence + '%';
                    confidenceFill.textContent = confidence.toFixed(1) + '%';
                    
                    document.querySelector('.classification-result').style.background = 
                        `linear-gradient(135deg, ${resultColor} 0%, #667eea 100%)`;
                    
                    results.style.display = 'block';
                } else {
                    showError(data.error || 'Unexpected error occurred');
                }
            } catch (error) {
                showError('Failed to connect to server: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function showError(message) {
            errorMessage.textContent = '❌ ' + message;
            errorMessage.style.display = 'block';
        }
        
        function resetUpload() {
            fileInput.value = '';
            fileName.textContent = '';
            results.style.display = 'none';
            errorMessage.style.display = 'none';
            originalImage.src = '';
            maskImage.src = '';
        }
    </script>
</body>
</html>
    '''
#-------------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint - returns segmentation mask with confidence scores.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
        
        # Read file bytes
        image_bytes = file.read()
        
        # Prepare image for model
        input_tensor = prepare_image(image_bytes, file.filename)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_probs = torch.sigmoid(output)
            pred_mask = (pred_probs > 0.5).float()
        
        # Convert mask to image
        mask_pil = mask_to_image(pred_mask[0])
        mask_base64 = encode_image_to_base64(mask_pil)
        
        # Calculate statistics
        water_pixels = pred_mask.sum().item()
        total_pixels = pred_mask.numel()
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Calculate approximate area (30m per pixel)
        pixel_area_km2 = (30 * 30) / 1_000_000  # 30m x 30m = 900m² = 0.0009 km²
        area_km2 = water_pixels * pixel_area_km2
        
        response = {
            'success': True,
            'water_percentage': round(water_percentage, 2),
            'area_km2': round(area_km2, 2),
            'water_pixels': int(water_pixels),
            'total_pixels': int(total_pixels),
            'mask_image': f"data:image/png;base64,{mask_base64}",
            'filename': file.filename
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== RUN APP ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
