# =============================================================================
# convert_model.py — Export fish_classifier.h5 to TFLite and TensorFlow.js
# =============================================================================
# Run this AFTER train_fish_classifier.py has produced fish_classifier.h5
#
# OUTPUTS:
#   fish_classifier.tflite       ← for Android / iOS / Raspberry Pi
#   web/model/model.json         ← for the browser (TensorFlow.js)
#   web/model/group1-shard1of1.bin
#
# REQUIREMENTS:
#   pip install tensorflow tensorflowjs
#
# USAGE:
#   python convert_model.py
# =============================================================================

import os
import sys
import json
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only for conversion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H5_PATH      = "fish_classifier.h5"
TFLITE_PATH  = "fish_classifier.tflite"
TFJS_OUT_DIR = Path("web") / "model"
CLASS_MAP    = "class_indices.json"

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

if not os.path.exists(H5_PATH):
    print(f"\n[ERROR] '{H5_PATH}' not found.")
    print("        Run train_fish_classifier.py first.\n")
    sys.exit(1)

if not os.path.exists(CLASS_MAP):
    print(f"[WARN] '{CLASS_MAP}' not found — web app will use generic labels.")

# =============================================================================
# 1. LOAD THE KERAS MODEL
# =============================================================================

print(f"\n[1/3] Loading model from '{H5_PATH}'...")

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(H5_PATH)
print(f"   Input  shape : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")
print(f"   Parameters   : {model.count_params():,}")

# =============================================================================
# 2. CONVERT TO TFLITE (for mobile / embedded)
# =============================================================================
# TFLite is a compressed, optimised format for running models on phones,
# Raspberry Pi, and other resource-constrained devices.
# We use INT8 dynamic-range quantisation: reduces model size ~4× and speeds
# up inference on ARM CPUs with minimal accuracy loss.

print("\n[2/3] Converting to TFLite (with INT8 quantisation)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic-range quantisation: weights compressed to int8, activations float32
# Great balance of speed/size/accuracy for a laptop CPU or mobile phone.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"   TFLite model saved -> {TFLITE_PATH}  ({size_kb:.0f} KB)")

# Quick inference test to confirm the TFLite model works
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
in_details  = interpreter.get_input_details()
out_details = interpreter.get_output_details()

import numpy as np
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
interpreter.set_tensor(in_details[0]["index"], dummy)
interpreter.invoke()
out = interpreter.get_tensor(out_details[0]["index"])
print(f"   TFLite smoke test passed — output shape: {out.shape}")

# =============================================================================
# 3. CONVERT TO TENSORFLOW.JS (for the browser web app)
# =============================================================================
# tensorflowjs_converter transforms the Keras H5 file into:
#   model.json        — architecture + weight manifest
#   *.bin files       — the actual weight values
# These are loaded directly by the browser using tfjs.

print("\n[3/3] Converting to TensorFlow.js format...")

try:
    import tensorflowjs as tfjs
    TFJS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(TFJS_OUT_DIR))
    print(f"   TF.js model saved  -> {TFJS_OUT_DIR}/")
    files = list(TFJS_OUT_DIR.iterdir())
    for f in files:
        print(f"     {f.name}  ({f.stat().st_size / 1024:.0f} KB)")

    # Auto-patch Keras 3 topology for TF.js 4.x compatibility
    fix_script = Path("fix_model_json.py")
    if fix_script.exists():
        print("\n   Patching model.json for TF.js compatibility...")
        exec(open(str(fix_script), encoding="utf-8").read())
    else:
        print("\n   [WARN] fix_model_json.py not found -- model.json may not load in browser.")
        print("          Run: py fix_model_json.py")

except ImportError:
    # tensorflowjs not installed — provide the manual CLI command instead
    print("\n   [INFO] tensorflowjs package not installed.")
    print("   Install it and run the conversion manually:\n")
    print("     pip install tensorflowjs\n")
    print("     tensorflowjs_converter \\")
    print(f"       --input_format=keras \\")
    print(f"       {H5_PATH} \\")
    print(f"       {TFJS_OUT_DIR}\n")
    print("   Then open web/index.html in a browser.")

# =============================================================================
# COPY CLASS INDEX MAP TO WEB FOLDER
# =============================================================================

if os.path.exists(CLASS_MAP):
    web_dir = Path("web")
    web_dir.mkdir(exist_ok=True)
    shutil.copy(CLASS_MAP, web_dir / "class_indices.json")
    print(f"\n   Class indices copied -> web/class_indices.json")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 55)
print("  CONVERSION COMPLETE")
print("=" * 55)
print(f"  TFLite  : {TFLITE_PATH}")
print(f"  TF.js   : {TFJS_OUT_DIR}/model.json")
print("=" * 55)
print("\nNext steps:")
print("  Mobile  -> copy fish_classifier.tflite to your Android/iOS project")
print("  Browser -> open web/index.html  (no server needed, works locally)\n")
