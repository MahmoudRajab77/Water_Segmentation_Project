# 🌊 Water Segmentation using Multispectral and Optical Data

## 📋 Project Overview

This project implements deep learning models for **water body segmentation** using multispectral and optical satellite data. The goal is to accurately identify water bodies for applications in flood management, water resource monitoring, and environmental conservation.

### 🎯 Key Features
- **Multispectral Data**: 12-channel satellite images (Coastal aerosol to Water occurrence)
- **Binary Segmentation**: Pixel-wise water classification
- **Two Model Variants**: From-scratch U-Net and Pretrained U-Net (ResNet34)
- **Comprehensive Evaluation**: IoU, Precision, Recall, F1-score
- **Data Visualization**: All 12 bands visualized individually

## 📊 Dataset

### Input Specifications
| Property | Value |
|----------|-------|
| **Image Size** | 128 × 128 pixels |
| **Channels** | 12 bands |
| **Ground Sampling** | 30m/pixel |
| **Format** | .tif (images), .png (masks) |

### Band Order (12 channels)
| # | Band Name | Description |
|---|-----------|-------------|
| 1 | Coastal aerosol | Aerosol detection, water quality |
| 2 | Blue | Visible blue light |
| 3 | Green | Visible green light |
| 4 | Red | Visible red light |
| 5 | NIR | Near Infrared |
| 6 | SWIR1 | Short Wave Infrared 1 |
| 7 | SWIR2 | Short Wave Infrared 2 |
| 8 | QA Band | Quality Assessment |
| 9 | Merit DEM | Digital Elevation Model |
| 10 | Copernicus DEM | Digital Elevation Model |
| 11 | ESA world cover | Land cover classification |
| 12 | Water occurrence | Historical water probability |

### Data Split
- **Training**: 214 samples (70%)
- **Validation**: 46 samples (15%)
- **Test**: 46 samples (15%)
- **Total**: 306 paired image-mask samples

## 🏗️ Project Structure
Water_Segmentation_Project/
│
├── FromScratch_Model/                          # Custom U-Net from scratch

│   ├── src/

│   │   ├── __init__.py

│   │   ├── data_load.py                        # Dataset class & splits

│   │   ├── model.py                             # U-Net architecture

│   │   ├── train.py                              # Training loop

│   │   └── visualize.py                           # Band visualization

│   ├── Results/

│   │   ├── all_bands_grid.png

│   │   ├── band_01_Coastal_aerosol.png

│   │   ├── band_02_Blue.png

│   │   ├── ...

│   │   ├── rgb_natural.png

│   │   ├── rgb_false_color.png

│   │   ├── water_indices.png

│   │   ├── test_results.txt

│   │   └── training_curves.png

│   ├── main.py

│   └── requirements.txt

│
 
├── PreTrained_Model/                           # U-Net with EfficientNet-b4 encoder

│   ├── src/

│   │   ├── __init__.py

│   │   ├── data_load.py                         

│   │   ├── model.py                              # Pretrained U-Net

│   │   ├── train.py                               # Training with pretrained

│   │   └── visualize.py                            # Band visualization

│   ├── Results/

│   │   ├── test_results.txt

│   │   └── training_curves.png

│   ├── APP/

│   │   ├── templates/
        
│   │   │   └── index.html
            
│   │   ├── static/
        
│   │   │   ├── script.js
            
│   │   │   └── style.css
            
│   │   └── app.py 

│   ├── main.py
        
│   └── requirements.txt

└── README.md


## 🚀 Installation & Setup
# Clone repository
git clone https://github.com/MahmoudRajab77/Water_Segmentation_Project.git
cd Water_Segmentation_Project

# Install dependencies
pip install -r requirements.txt

# For FromScratch model
cd FromScratch_Model
python main.py 

# For Pretrained model
cd PreTrained_Model
python main.py 
