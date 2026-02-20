# Philippine Fish Classifier - Complete Guide

## 🎯 Overview

This project is a **Philippine fish species classifier** that uses:
- **MobileNetV2** - A lightweight CNN pre-trained on ImageNet
- **Transfer Learning** - We freeze the base model and train only the top layers
- **TensorFlow.js** - For browser-based inference (no server needed)

### Supported Species
| Filipino Name | English Name | Class Index |
|---------------|--------------|-------------|
| Bangus | Milkfish | 0 |
| Tilapia | Tilapia | 1 |
| Galunggong | Round Scad | 2 |

---

## 📁 Project Structure

```
FINE TUNED/
├── dataset/
│   ├── train/
│   │   ├── Bangus/          # Training images for Bangus
│   │   ├── Tilapia/         # Training images for Tilapia
│   │   └── Galunggong/      # Training images for Galunggong
│   └── val/
│       ├── Bangus/          # Validation images
│       ├── Tilapia/
│       └── Galunggong/
├── web/
│   ├── index.html           # Web UI for classification
│   ├── class_indices.json   # Maps class names to indices
│   └── model/
│       ├── model.json       # TF.js model topology
│       └── group1-shard*.bin # Model weights (binary)
├── train_fish_classifier.py  # Training script
├── convert_model.py          # Converts .h5 to TFLite + TF.js
├── fix_model_json.py         # Patches Keras 3 JSON for TF.js
├── fish_classifier.h5        # Trained Keras model
├── fish_classifier.tflite    # Mobile-optimized model
└── requirements.txt          # Python dependencies
```

---

## 🔧 How It Works

### 1. Training Pipeline

```
┌─────────────────┐
│  Dataset Images │
│  (224×224 RGB)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Data Aug.     │  ← Random flip, rotation, zoom, brightness
│   (Training)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MobileNetV2    │  ← Pre-trained ImageNet weights (FROZEN)
│  (Feature Ext.) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Custom Head    │  ← GlobalAvgPool → BN → Dense(128) → Dropout → Dense(3)
│  (TRAINABLE)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Softmax       │  ← Outputs probabilities for each class
│   Output        │
└─────────────────┘
```

### 2. Model Architecture

| Layer | Output Shape | Parameters | Trainable |
|-------|-------------|------------|-----------|
| MobileNetV2 | (7, 7, 1280) | 2.2M | ❌ Frozen |
| GlobalAveragePooling2D | (1280,) | 0 | - |
| BatchNormalization | (1280,) | 5,120 | ✅ |
| Dense (ReLU) | (128,) | 163,968 | ✅ |
| Dropout (0.4) | (128,) | 0 | - |
| Dense (Softmax) | (3,) | 387 | ✅ |

**Total: ~2.6M parameters, ~170K trainable**

### 3. Conversion Pipeline

```
fish_classifier.h5
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
fish_classifier.tflite    web/model/model.json
(Mobile: Android/iOS)     (Browser: TF.js)
                               │
                               ▼
                          fix_model_json.py
                          (Keras 3 → TF.js compat)
```

---

## 🚀 Step-by-Step: Training & Retraining

### Prerequisites

```powershell
# Install Python 3.10+ if not already installed
winget install Python.Python.3.10

# Install dependencies
py -m pip install tensorflow pillow
```

### Step 1: Prepare Your Dataset

1. Create folder structure:
   ```
   dataset/
   ├── train/
   │   ├── Bangus/      (put ~20-50+ images)
   │   ├── Tilapia/     (put ~20-50+ images)
   │   └── Galunggong/  (put ~20-50+ images)
   └── val/
       ├── Bangus/      (put ~5-10 images)
       ├── Tilapia/     (put ~5-10 images)
       └── Galunggong/  (put ~5-10 images)
   ```

2. Image requirements:
   - Format: JPG, PNG, or JPEG
   - Size: Any (will be resized to 224×224)
   - Quality: Clear photos of the fish
   - Variety: Different angles, lighting, backgrounds

### Step 2: Train the Model

```powershell
cd "C:\Users\Name\Desktop\FINE TUNED"
py train_fish_classifier.py
```

**What happens:**
- Loads images from `dataset/train/` and `dataset/val/`
- Applies data augmentation (flip, rotate, zoom)
- Trains for 10 epochs with early stopping
- Saves best model to `fish_classifier.h5`
- Saves class mapping to `class_indices.json`

**Expected output:**
```
Found 57 images in 3 classes
Epoch 1/10 - loss: 1.2 - accuracy: 0.40 - val_accuracy: 0.35
Epoch 2/10 - loss: 0.8 - accuracy: 0.55 - val_accuracy: 0.45
...
Model saved to fish_classifier.h5
```

### Step 3: Convert for Web/Mobile

```powershell
py convert_model.py
```

**What happens:**
- Loads `fish_classifier.h5`
- Creates `fish_classifier.tflite` (for mobile apps)
- Creates `web/model/model.json` + weight shards (for browser)
- Auto-patches JSON for TF.js compatibility

### Step 4: Test in Browser

```powershell
# Option A: Python HTTP server
cd web
py -m http.server 8080
# Open http://localhost:8080

# Option B: VS Code Live Server
# Right-click index.html → Open with Live Server
```

---

## 🔄 How to Retrain After Changing Dataset

### Quick Retrain (Same Classes)

```powershell
# 1. Delete old model (optional but recommended)
del fish_classifier.h5

# 2. Retrain
py train_fish_classifier.py

# 3. Convert to TF.js
py convert_model.py

# 4. Refresh browser (Ctrl+Shift+R)
```

### Adding New Fish Species

1. Create new folders in `dataset/train/` and `dataset/val/`
2. Add images to the new folders
3. Update `SPECIES_META` in `web/index.html`:
   ```javascript
   const SPECIES_META = {
     Bangus:      { icon: "🐟", englishName: "Milkfish" },
     Tilapia:     { icon: "🐠", englishName: "Tilapia" },
     Galunggong:  { icon: "🐡", englishName: "Round Scad" },
     NewFish:     { icon: "🐟", englishName: "New Fish Name" },  // Add this
   };
   ```
4. Retrain: `py train_fish_classifier.py`
5. Convert: `py convert_model.py`

---

## 📊 Improving Accuracy

### More Data
- **Minimum**: 20 images per class
- **Recommended**: 100+ images per class
- **Best**: 500+ images per class

### Better Data
- Clear, well-lit photos
- Various angles (side, top, whole fish, fillet)
- Different backgrounds
- Both raw and cooked (if applicable)

### Training Tips
Edit `train_fish_classifier.py`:

```python
# Increase epochs (line ~50)
EPOCHS = 20  # instead of 10

# Lower learning rate for fine-tuning (line ~80)
lr=1e-4  # instead of 1e-3

# Unfreeze last few layers of MobileNet (add after line ~70)
for layer in base_model.layers[-20:]:
    layer.trainable = True
```

---

## 🐛 Troubleshooting

### "Cannot load model" in browser
```powershell
# Regenerate and fix model.json
py convert_model.py
# Hard refresh browser: Ctrl+Shift+R
```

### Low accuracy
1. Add more diverse training images
2. Check for mislabeled images
3. Increase training epochs
4. Try unfreezing more MobileNet layers

### Training crashes / Out of memory
```python
# Reduce batch size in train_fish_classifier.py
BATCH_SIZE = 8  # instead of 16
```

### Unicode errors on Windows
```powershell
py -X utf8 train_fish_classifier.py
py -X utf8 convert_model.py
```

---

## 📱 Deploying to Mobile

### Android (TFLite)
1. Copy `fish_classifier.tflite` to `app/src/main/assets/`
2. Copy `class_indices.json` to assets
3. Use TFLite Interpreter in your app

### iOS (Core ML)
```bash
# Convert to Core ML (requires coremltools)
pip install coremltools
python -c "
import coremltools as ct
import tensorflow as tf
model = tf.keras.models.load_model('fish_classifier.h5')
mlmodel = ct.convert(model)
mlmodel.save('FishClassifier.mlmodel')
"
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `train_fish_classifier.py` | Main training script |
| `convert_model.py` | Converts .h5 → TFLite + TF.js |
| `fix_model_json.py` | Patches Keras 3 JSON for TF.js |
| `write_tfjs.py` | Alternative TF.js converter |
| `download_dataset.py` | Downloads sample images from web |
| `fish_classifier.h5` | Trained Keras model (25 MB) |
| `fish_classifier.tflite` | Mobile model (2.7 MB) |
| `web/index.html` | Browser UI |
| `web/model/model.json` | TF.js model topology |
| `web/model/group1-shard*.bin` | TF.js weights (~10 MB total) |

---

## ✅ Quick Reference Commands

```powershell
# Train model
py train_fish_classifier.py

# Convert to web/mobile formats
py convert_model.py

# Start local server
cd web && py -m http.server 8080

# Fix model.json manually (if needed)
py -X utf8 fix_model_json.py

# Validate model.json
py validate_model.py

# Check dataset
Get-ChildItem dataset\train -Recurse | Measure-Object
```

---

*Created for Philippine Fish Classifier v1.0*
