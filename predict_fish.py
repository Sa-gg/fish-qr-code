# =============================================================================
# predict_fish.py — Run a single image through the trained classifier
# =============================================================================
# USAGE:
#   python predict_fish.py path/to/your/image.jpg
#
# EXAMPLE:
#   python predict_fish.py test_photo.jpg
# =============================================================================

import os
import sys
import json
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only

import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH      = "fish_classifier.h5"
CLASS_MAP_PATH  = "class_indices.json"   # written automatically by training script
IMG_HEIGHT      = 224
IMG_WIDTH       = 224

# Load class names from the JSON file saved during training.
# Format: {"Bangus": 0, "Galunggong": 1, "Tilapia": 2}
# Sorted by index so the list order matches the model's output neurons.
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH) as f:
        _idx_map = json.load(f)
    CLASS_NAMES = [k for k, _ in sorted(_idx_map.items(), key=lambda x: x[1])]
else:
    CLASS_NAMES = ["Bangus", "Galunggong", "Tilapia"]   # fallback


def predict(img_path: str) -> None:
    """Load the model and predict the species for a single image."""

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
        print("        Run train_fish_classifier.py first to create it.")
        sys.exit(1)

    if not os.path.exists(img_path):
        print(f"[ERROR] Image file '{img_path}' not found.")
        sys.exit(1)

    # Load model
    print(f"Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    # Load and preprocess the image exactly as during training
    img = Image.open(img_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0  # [0,1], shape (224,224,3)
    img_array = np.expand_dims(img_array, 0)              # add batch dim → (1,224,224,3)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (NUM_CLASSES,)

    # Display results
    print(f"\nPrediction results for: {img_path}")
    print("-" * 40)
    for name, prob in sorted(zip(CLASS_NAMES, predictions), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:<15} {prob * 100:5.1f}%  {bar}")
    print("-" * 40)

    top_idx   = int(np.argmax(predictions))
    top_label = CLASS_NAMES[top_idx]
    top_conf  = predictions[top_idx] * 100

    print(f"\n  Result  : {top_label}")
    print(f"  Confidence: {top_conf:.1f}%")

    if top_conf < 60:
        print("  [WARNING] Low confidence — the model is uncertain.")
        print("            Consider adding more training images for this species.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_fish.py <path_to_image>")
        print("Example: python predict_fish.py my_fish_photo.jpg")
        sys.exit(1)

    predict(sys.argv[1])
