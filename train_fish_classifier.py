# =============================================================================
# train_fish_classifier.py — Philippine Fish Species Classifier
# =============================================================================
# Hardware target: AMD A8-7680 CPU + AMD Radeon R7 integrated graphics, 12 GB RAM
#
# WORKFLOW:
#   Step 1 → download_dataset.py        (pull images automatically)
#   Step 2 → train_fish_classifier.py   (YOU ARE HERE — train the model)
#   Step 3 → convert_model.py           (export to TFLite + TensorFlow.js)
#   Step 4 → open web/index.html        (test in browser)
#
# HOW TO ADD MORE SPECIES LATER:
#   1. Add a folder  dataset/train/<NewSpecies>/     (10+ images)
#   2. Add a folder  dataset/validation/<NewSpecies>/ (3-5 images)
#   3. Re-run this script — zero code changes needed.
# =============================================================================

import os
import sys
import json

# ---------------------------------------------------------------------------
# GPU / HARDWARE SETUP
#
# Option A (default): Pure CPU — works on ANY machine, no driver needed.
# Option B: tensorflow-directml — enables AMD/Intel integrated GPU on Windows.
#           Install with:  pip install tensorflow-directml
#           Then set USE_DIRECTML = True below.
# ---------------------------------------------------------------------------
USE_DIRECTML = False   # set True if you installed tensorflow-directml

if not USE_DIRECTML:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # hide all GPUs
else:
    print("[INFO] DirectML mode — will attempt AMD Radeon R7.")

# Limit threads to match AMD A8-7680's 4 cores, reducing context-switching overhead
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"]  = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all machines
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
# NOTE: ImageDataGenerator was removed in Keras 3 (TF 2.16+).
# We now use the modern tf.keras.utils.image_dataset_from_directory + tf.data.

print(f"TensorFlow version : {tf.__version__}")
print(f"Devices visible    : {[d.name for d in tf.config.list_physical_devices()]}")

# =============================================================================
# CONFIGURATION — edit these values to tune training
# =============================================================================

TRAIN_DIR      = "dataset/train"
VAL_DIR        = "dataset/validation"
MODEL_SAVE_H5  = "fish_classifier.h5"      # Keras format (used by convert_model.py)

IMG_HEIGHT = 224    # MobileNetV2 standard input size
IMG_WIDTH  = 224
CHANNELS   = 3

# ---- Hyper-parameters tuned for AMD A8-7680, 12 GB RAM --------------------
BATCH_SIZE        = 8     # small batch = less RAM, CPU-friendly
INITIAL_EPOCHS    = 20    # Phase 1: train only the new head
FINE_TUNE_EPOCHS  = 15    # Phase 2: optional fine-tuning
LEARNING_RATE     = 1e-3
FINE_TUNE_LR      = 5e-6  # must be very small — protects pretrained weights

# MobileNetV2 has 154 layers. 30 = last ~2 Inverted Residual blocks.
# Set to 0 to skip fine-tuning (faster but lower accuracy).
FINE_TUNE_AT = 30

L2_FACTOR = 1e-4    # weight decay to reduce overfitting on small datasets

# =============================================================================
# STEP 1 — VALIDATE DATASET
# =============================================================================

def validate_dataset(train_dir, val_dir):
    """Abort early with a helpful message if folders or images are missing."""
    for split_name, split_path in [("train", train_dir), ("validation", val_dir)]:
        if not os.path.isdir(split_path):
            print(f"\n[ERROR] '{split_path}' not found.")
            print("        Run:  python download_dataset.py\n")
            sys.exit(1)
        species_dirs = [d for d in os.listdir(split_path)
                        if os.path.isdir(os.path.join(split_path, d))]
        if not species_dirs:
            print(f"\n[ERROR] No species subfolders in '{split_path}'.")
            sys.exit(1)
        for sp in species_dirs:
            imgs = [f for f in os.listdir(os.path.join(split_path, sp))
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
            if len(imgs) < 2:
                print(f"   [WARN] {split_name}/{sp}: only {len(imgs)} image(s) — add more.")

print("\n[1/6] Validating dataset folders...")
validate_dataset(TRAIN_DIR, VAL_DIR)
print("      OK.")

# =============================================================================
# STEP 2 — DATA LOADING WITH AUGMENTATION (Keras 3 / TF 2.16+ compatible)
# =============================================================================
# ImageDataGenerator was removed in Keras 3.  We now use:
#   • image_dataset_from_directory  — loads images from folder structure
#   • tf.data pipeline              — fast, memory-efficient streaming
#   • Keras augmentation layers     — same transforms, now inside the graph

print("\n[2/6] Setting up data pipeline...")

# Load raw datasets — subfolder name = class label (alphabetical order)
train_ds_raw = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",   # one-hot vectors (e.g. [1,0,0] for Bangus)
    shuffle=True,
    seed=42
)

val_ds_raw = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
    seed=42
)

# Class names come from the subfolder names, sorted alphabetically
CLASS_NAMES = train_ds_raw.class_names
NUM_CLASSES = len(CLASS_NAMES)

# Count total images by iterating batches once
train_total = sum(labels.shape[0] for _, labels in train_ds_raw)
val_total   = sum(labels.shape[0] for _, labels in val_ds_raw)

print(f"   Species detected  : {CLASS_NAMES}")
print(f"   Training images   : {train_total}")
print(f"   Validation images : {val_total}")

# Save class index mapping (index = alphabetical position, matches model output)
class_indices = {name: idx for idx, name in enumerate(CLASS_NAMES)}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print("   Class map saved   → class_indices.json")

# ---- Preprocessing & Augmentation layers ----
# Rescaling converts pixel values from [0,255] → [0,1] as MobileNetV2 expects.
# Augmentation layers apply random transforms only during training,
# simulating a larger dataset from your 19 images per species.

rescale = layers.Rescaling(1.0 / 255.0, name="rescaling")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),               # mirror left-right
    layers.RandomRotation(30 / 360),               # ±30 degree rotation
    layers.RandomZoom(0.25),                       # zoom in/out ±25%
    layers.RandomTranslation(0.2, 0.2),            # shift ±20%
    layers.RandomBrightness(factor=0.4),           # darker ↔ brighter
    layers.RandomContrast(factor=0.2),             # vary contrast
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_train(images, labels):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    images = data_augmentation(images, training=True)
    return images, labels

def preprocess_val(images, labels):
    images = tf.cast(images, tf.float32)
    images = rescale(images)
    return images, labels

# Build final pipelines with parallel preprocessing and background prefetch
train_gen = train_ds_raw.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_gen   = val_ds_raw.map(preprocess_val,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# =============================================================================
# STEP 3 — BUILD MODEL (MobileNetV2 Transfer Learning)
# =============================================================================
# MobileNetV2 improvements over original MobileNet:
#   • Inverted residuals + linear bottlenecks → better accuracy at same size
#   • 3.4M parameters — ideal for CPU inference and mobile deployment
#
# Architecture:
#   Input (224×224×3)
#     → MobileNetV2 feature extractor  [FROZEN in Phase 1]
#     → GlobalAveragePooling2D          7×7×1280 → 1280-vector
#     → BatchNormalization              stabilises small-batch training
#     → Dense(256, relu, L2)            new trainable classifier head
#     → Dropout(0.5)                    prevents overfitting
#     → Dense(NUM_CLASSES, softmax)     final species probabilities

print("\n[3/6] Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
    include_top=False,      # drop ImageNet classifier — we add our own
    weights="imagenet"      # use pretrained weights (~14 MB download, once)
)
base_model.trainable = False  # freeze ALL base layers for Phase 1

inputs  = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="image_input")
x       = base_model(inputs, training=False)   # training=False keeps BN layers in inference mode
x       = layers.GlobalAveragePooling2D(name="gap")(x)
x       = layers.BatchNormalization(name="head_bn")(x)
x       = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(L2_FACTOR),
                       name="head_dense")(x)
x       = layers.Dropout(0.5, name="head_dropout")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

model = keras.Model(inputs, outputs, name="fish_classifier")

trainable     = sum(np.prod(v.shape) for v in model.trainable_variables)
non_trainable = sum(np.prod(v.shape) for v in model.non_trainable_variables)
print(f"   Trainable params     : {trainable:,}")
print(f"   Non-trainable params : {non_trainable:,}  (frozen MobileNetV2)")

# =============================================================================
# STEP 4 — PHASE 1: TRAIN CLASSIFIER HEAD
# =============================================================================
# Only the Dense layers we added are updated.
# MobileNetV2 frozen → fast on A8-7680 CPU.

print(f"\n[4/6] Phase 1 — Training classifier head ({INITIAL_EPOCHS} epochs max)...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("checkpoints", exist_ok=True)

callbacks_p1 = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "checkpoints/phase1_best.h5",
        monitor="val_accuracy", save_best_only=True, verbose=0
    ),
]

history1 = model.fit(
    train_gen,
    epochs=INITIAL_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_p1,
    verbose=1
)
print(f"\n   Phase 1 best val accuracy: {max(history1.history['val_accuracy'])*100:.1f}%")

# =============================================================================
# STEP 5 — PHASE 2: FINE-TUNING (optional)
# =============================================================================
# Unfreeze the last FINE_TUNE_AT layers of MobileNetV2 and retrain with a
# very small learning rate.  This adapts high-level feature detectors to fish.
#
# WHY LOW LR: Large learning rate would "forget" ImageNet knowledge → worse results.
# BatchNorm layers in the base stay frozen — essential for small batch sizes.

if FINE_TUNE_AT > 0 and train_total >= 30:
    print(f"\n[5/6] Phase 2 — Fine-tuning last {FINE_TUNE_AT} MobileNetV2 layers "
          f"({FINE_TUNE_EPOCHS} epochs max)...")

    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False
    # Keep ALL BatchNorm layers frozen — critical for stability with batch_size=8
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    ft_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"   Trainable params now : {ft_params:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-8, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "checkpoints/phase2_best.h5",
            monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    history2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        verbose=1
    )
    print(f"\n   Phase 2 best val accuracy: {max(history2.history['val_accuracy'])*100:.1f}%")

    merged       = {k: history1.history[k] + history2.history[k] for k in history1.history}
    phase2_start = len(history1.history["accuracy"])

else:
    print(f"\n[5/6] Skipping Phase 2 (FINE_TUNE_AT={FINE_TUNE_AT}, samples={train_total}).")
    merged       = history1.history
    phase2_start = None

# =============================================================================
# STEP 6 — SAVE MODEL
# =============================================================================

print(f"\n[6/6] Saving model...")
model.save(MODEL_SAVE_H5)
print(f"   Keras H5 saved  → {MODEL_SAVE_H5}")
print(f"   Next step       → python convert_model.py")

# =============================================================================
# TRAINING CURVES
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Philippine Fish Classifier — Training Results", fontsize=13, fontweight="bold")
ep = range(len(merged["accuracy"]))

for ax, metric, title in [
    (axes[0], "accuracy", "Accuracy"),
    (axes[1], "loss",     "Loss"),
]:
    ax.plot(ep, merged[metric],           label=f"Train {title}", color="#2196F3", linewidth=2)
    ax.plot(ep, merged[f"val_{metric}"],  label=f"Val {title}",   color="#FF9800", linewidth=2)
    if phase2_start:
        ax.axvline(phase2_start, color="#9C27B0", linestyle="--",
                   linewidth=1.5, label="Fine-tune start")
    ax.set_title(title) ; ax.set_xlabel("Epoch") ; ax.set_ylabel(title)
    ax.legend() ; ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("   Training chart  → training_curves.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print("\n" + "=" * 50)
print("  TRAINING COMPLETE")
print("=" * 50)
print(f"  Species      : {CLASS_NAMES}")
print(f"  Val Accuracy : {val_acc * 100:.1f}%")
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Model saved  : {MODEL_SAVE_H5}")
print("=" * 50)
print("\nNext steps:")
print("  1. python convert_model.py   ← export to TFLite + TF.js")
print("  2. Open web/index.html       ← test in browser\n")
