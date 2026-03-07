// DOM Elements
const imageInput = document.getElementById('imageInput');
const maskInput = document.getElementById('maskInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const imageName = document.getElementById('imageName');
const maskName = document.getElementById('maskName');
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
const classificationResult = document.getElementById('classificationResult');
const metricsCard = document.getElementById('metricsCard');
const iouValue = document.getElementById('iouValue');
const precisionValue = document.getElementById('precisionValue');
const recallValue = document.getElementById('recallValue');
const f1Value = document.getElementById('f1Value');

// State
let groundTruthFile = null;
let imageFile = null;

// Disable analyze button initially
analyzeBtn.disabled = true;

// Event Listeners
imageInput.addEventListener('change', function(e) {
    imageFile = e.target.files[0];
    if (!imageFile) return;
    
    imageName.textContent = `📁 Image: ${imageFile.name}`;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage.src = e.target.result;
    };
    reader.readAsDataURL(imageFile);
    
    analyzeBtn.disabled = false;
});

maskInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    groundTruthFile = file;
    maskName.textContent = `📋 Mask: ${file.name}`;
});

analyzeBtn.addEventListener('click', startAnalysis);

// Functions
function startAnalysis() {
    if (!imageFile) {
        showError('Please select an image first');
        return;
    }
    uploadAndAnalyze(imageFile, groundTruthFile);
}

async function uploadAndAnalyze(imageFile, maskFile = null) {
    results.style.display = 'none';
    metricsCard.style.display = 'none';
    errorMessage.style.display = 'none';
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', imageFile);
    if (maskFile) {
        formData.append('mask', maskFile);
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Display mask
            maskImage.src = data.mask_image;
            if (data.original_image) {
                originalImage.src = data.original_image;
            }
            
            // Update basic stats
            waterPercentage.textContent = data.water_percentage + '%';
            areaValue.textContent = data.area_km2 + ' km²';
            pixelCount.textContent = data.water_pixels.toLocaleString();
            pixelInfo.textContent = `Total pixels: ${data.total_pixels} | Water pixels: ${data.water_pixels}`;
            
            // Update confidence bar
            const confidence = data.avg_confidence;
            confidenceFill.style.width = confidence + '%';
            confidenceFill.textContent = confidence.toFixed(1) + '%';
            confidenceDetail.textContent = `Average model confidence: ${confidence.toFixed(1)}%`;
            
            // Update classification text
            const coverage = data.water_percentage;
            let resultText = '';
            let resultColor = '';
            
            if (coverage >= 70) {
                resultText = '🌊 High Water Concentration';
                resultColor = '#43e97b';
            } else if (coverage >= 30) {
                resultText = '💧 Moderate Water Presence';
                resultColor = '#f9d423';
            } else {
                resultText = '🏝️ Low Water / Mostly Land';
                resultColor = '#fa709a';
            }
            
            classificationText.textContent = resultText;
            classificationResult.style.background = 
                `linear-gradient(135deg, ${resultColor} 0%, #667eea 100%)`;
            
            // If metrics are available
            if (data.iou !== undefined) {
                metricsCard.style.display = 'block';
                iouValue.textContent = data.iou;
                precisionValue.textContent = data.precision;
                recallValue.textContent = data.recall;
                f1Value.textContent = data.f1;
            }
            
            results.style.display = 'block';
        } else {
            showError(data.error || 'Unexpected error occurred');
        }
    } catch (error) {
        showError('Failed to connect to server: ' + error.message);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function showError(message) {
    errorMessage.textContent = '❌ ' + message;
    errorMessage.style.display = 'block';
}

function resetUpload() {
    imageInput.value = '';
    maskInput.value = '';
    imageFile = null;
    groundTruthFile = null;
    imageName.textContent = '';
    maskName.textContent = '';
    results.style.display = 'none';
    metricsCard.style.display = 'none';
    errorMessage.style.display = 'none';
    originalImage.src = '';
    maskImage.src = '';
    analyzeBtn.disabled = true;
}

// Make resetUpload available globally
window.resetUpload = resetUpload;
