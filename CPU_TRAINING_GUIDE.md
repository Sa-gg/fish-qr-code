# Training on CPU-Only Systems (No GPU Required)

This guide covers how to train the Philippine Fish Classifier on systems **without a dedicated GPU**, including:
- Desktops with integrated graphics (Intel UHD, AMD Radeon Vega)
- Laptops with disabled/broken dGPU
- Low-end machines

---

## 🖥️ System Requirements

### Minimum Specs
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Dual-core 2.0 GHz | Quad-core 2.5+ GHz |
| RAM | 4 GB | 8+ GB |
| Storage | 2 GB free | 5+ GB free |
| Python | 3.9+ | 3.10 |
| OS | Windows 10/11, Linux, macOS | Any |

### Your Current Setup (Example)
```
CPU: AMD A8-7680 (4 cores, 3.5 GHz)
RAM: 8 GB
GPU: Integrated AMD Radeon R7
Storage: SSD recommended
```

**This is perfectly fine for training!** MobileNetV2 is lightweight and designed for edge devices.

---

## ⚙️ Setup for CPU-Only Training

### Step 1: Install CPU-Optimized TensorFlow

TensorFlow automatically uses CPU when no GPU is detected. The version we're using (2.20.0) includes optimizations like oneDNN for faster CPU inference.

```powershell
# Already installed, but to verify:
py -m pip show tensorflow
# Should show: Version: 2.20.0
```

### Step 2: Configure TensorFlow for CPU

Add these environment variables to suppress GPU-related warnings:

```powershell
# PowerShell (run before training)
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:TF_ENABLE_ONEDNN_OPTS = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"

# Or in Python (at top of script)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Step 3: Verify CPU-Only Mode

```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
# Should print: GPUs: []

print("CPUs:", tf.config.list_physical_devices('CPU'))
# Should print: CPUs: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
```

---

## 🚀 Training Commands

### Basic Training (CPU)

```powershell
cd "C:\Users\Name\Desktop\FINE TUNED"

# Set environment variables
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:CUDA_VISIBLE_DEVICES = "-1"

# Run training
py train_fish_classifier.py
```

### With UTF-8 Encoding (Windows)

```powershell
py -X utf8 train_fish_classifier.py
```

---

## ⏱️ Expected Training Times (CPU)

| Dataset Size | Training Time | Notes |
|--------------|---------------|-------|
| 50 images | 5-10 min | Current dataset |
| 100 images | 10-20 min | Good starting point |
| 500 images | 30-60 min | Recommended for accuracy |
| 1000+ images | 1-2 hours | High accuracy |

*Times based on AMD A8-7680 / Intel i5-8th gen class CPUs*

---

## 🔧 Optimizations for Slow Systems

### 1. Reduce Batch Size

Edit `train_fish_classifier.py`:

```python
# Line ~30 - Lower batch size uses less RAM
BATCH_SIZE = 8  # Default is 16, try 8 or even 4
```

### 2. Reduce Image Size (Faster but Less Accurate)

```python
# Line ~25 - Smaller images = faster training
IMG_SIZE = 160  # Default is 224, try 160 or 128
```

### 3. Fewer Training Epochs

```python
# Line ~50 - Fewer epochs = faster completion
EPOCHS = 5  # Default is 10
```

### 4. Disable Data Augmentation (Not Recommended)

```python
# In train_fish_classifier.py, simplify augmentation:
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    # Remove other augmentation parameters
)
```

---

## 💻 Training on Laptop with Integrated Graphics

### Disable Broken dGPU (If Not Already Done)

1. **Device Manager** → Display adapters
2. Right-click broken GPU → **Disable device**
3. Restart laptop

### Power Settings for Training

1. **Control Panel** → Power Options
2. Select **High Performance** plan
3. Or create custom plan:
   - Processor power: 100% min/max
   - Turn off display: Never (during training)
   - Sleep: Never (during training)

### Prevent Thermal Throttling

- Use laptop on hard, flat surface
- Clean dust from vents
- Consider a cooling pad
- Monitor temps with HWiNFO64

### Check CPU Temperature

```powershell
# Install OpenHardwareMonitor or use:
Get-WmiObject -Namespace "root\wmi" -Class MSAcpi_ThermalZoneTemperature |
    Select-Object InstanceName, @{n='Temp';e={($_.CurrentTemperature - 2732) / 10}}
```

---

## 📊 Monitoring Training Progress

### Watch Memory Usage

```powershell
# In separate PowerShell window
while ($true) {
    $mem = Get-Process -Name python -ErrorAction SilentlyContinue |
           Measure-Object -Property WorkingSet64 -Sum
    Write-Host "Python RAM: $([math]::Round($mem.Sum / 1GB, 2)) GB"
    Start-Sleep 5
}
```

### Watch CPU Usage

```powershell
# Simple CPU monitor
while ($true) {
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time'
    Write-Host "CPU: $([math]::Round($cpu.CounterSamples.CookedValue, 1))%"
    Start-Sleep 2
}
```

---

## 🔄 Complete Workflow (CPU Training)

```powershell
# 1. Navigate to project
cd "C:\Users\Name\Desktop\FINE TUNED"

# 2. Set CPU-only mode
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

# 3. Check dataset
Get-ChildItem dataset\train -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -File).Count
    Write-Host "$($_.Name): $count images"
}

# 4. Train the model (5-15 minutes)
py -X utf8 train_fish_classifier.py

# 5. Convert to TF.js (1-2 minutes)
py -X utf8 convert_model.py

# 6. Validate
py validate_model.py

# 7. Test in browser
cd web
py -m http.server 8080
# Open http://localhost:8080
```

---

## ❓ Troubleshooting CPU Training

### "Out of Memory" Error

```python
# Reduce batch size
BATCH_SIZE = 4

# Or limit TensorFlow memory growth
import tensorflow as tf
tf.config.set_soft_device_placement(True)
```

### Training is Very Slow

1. Close other applications
2. Check for thermal throttling
3. Use smaller images (IMG_SIZE = 160)
4. Reduce batch size
5. Consider overnight training

### "No module named 'tensorflow'" 

```powershell
py -m pip install tensorflow
```

### High CPU Temperature (>90°C)

1. Stop training immediately
2. Let laptop cool down
3. Clean dust from vents
4. Reduce CPU power in BIOS or use throttlestop
5. Consider training overnight when cooler

---

## 🎯 Tips for Best Results on CPU

1. **Train overnight** - Let it run while you sleep
2. **Use SSD** - Faster data loading than HDD
3. **Close browsers** - Free up RAM
4. **Plug in laptop** - Don't rely on battery
5. **Good ventilation** - Prevent throttling
6. **More data > More epochs** - Adding images helps more than longer training

---

## 📈 Performance Comparison

| Hardware | Training Time (100 images) | Notes |
|----------|---------------------------|-------|
| NVIDIA RTX 3060 | 1-2 min | GPU accelerated |
| AMD Ryzen 5600X | 8-12 min | Modern CPU |
| Intel i5-8th Gen | 12-18 min | Laptop CPU |
| AMD A8-7680 | 15-25 min | Older desktop APU |
| Intel Celeron | 30-45 min | Budget CPU |

**GPU is faster, but CPU training works perfectly fine for small datasets!**

---

*This guide is part of the Philippine Fish Classifier project.*
