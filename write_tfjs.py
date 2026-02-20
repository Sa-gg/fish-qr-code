"""
write_tfjs.py — Manually writes a TF.js layers-model from a Keras .h5 file.
No tensorflowjs package needed.

Usage:  python write_tfjs.py
Output: web/model/model.json  +  web/model/group1-shard*.bin
"""
import os, sys, json, struct, shutil
import numpy as np

# Suppress TF noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

OUT_DIR  = "web/model"
H5_PATH  = "fish_classifier.h5"
IDX_PATH = "class_indices.json"

# ── 1. Load model ────────────────────────────────────────────────────────────
print("[1/4] Loading model ...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f"      {model.input_shape} -> {model.output_shape}")

# ── 2. Collect weights ───────────────────────────────────────────────────────
print("[2/4] Collecting weights ...")
weight_names   = []
weight_shapes  = []
weight_dtypes  = []
weight_arrays  = []

for w in model.weights:
    arr = w.numpy().astype(np.float32)
    weight_names.append(w.name)
    weight_shapes.append(list(arr.shape))
    weight_dtypes.append("float32")
    weight_arrays.append(arr.flatten().tobytes())

# ── 3. Write binary weight shards ────────────────────────────────────────────
print("[3/4] Writing binary weight shards ...")
os.makedirs(OUT_DIR, exist_ok=True)

raw = b"".join(weight_arrays)
SHARD = 4 * 1024 * 1024  # 4 MB per shard
shard_paths = []
for i in range(0, len(raw), SHARD):
    fname = f"group1-shard{i//SHARD + 1}of{(len(raw) + SHARD - 1)//SHARD}.bin"
    path  = os.path.join(OUT_DIR, fname)
    with open(path, "wb") as f:
        f.write(raw[i : i + SHARD])
    shard_paths.append(fname)
    print(f"      Wrote {fname}  ({os.path.getsize(path)//1024} KB)")

# ── 4. Build model.json ──────────────────────────────────────────────────────
print("[4/4] Writing model.json ...")

# Weight manifest entries
weight_manifest = []
offset = 0
for name_w, shape_w, dtype_w, arr_b in zip(
        weight_names, weight_shapes, weight_dtypes, weight_arrays):
    weight_manifest.append({
        "name":  name_w,
        "shape": shape_w,
        "dtype": dtype_w,
    })
    offset += len(arr_b)

manifest = [{
    "paths":   shard_paths,
    "weights": weight_manifest,
}]

model_topology = json.loads(model.to_json())


def patch_for_tfjs(obj):
    """Recursively patch Keras 3 JSON topology to TF.js / Keras 2 format."""
    if isinstance(obj, dict):
        # Fix InputLayer: Keras 3 uses 'batch_shape', TF.js needs 'batchInputShape'
        if obj.get("class_name") == "InputLayer" and "config" in obj:
            cfg = obj["config"]
            if "batch_shape" in cfg:
                cfg["batchInputShape"] = cfg.pop("batch_shape")
            if "input_shape" in cfg:
                cfg["inputShape"] = cfg.pop("input_shape")
            # TF.js requires dtype as simple string, not dict
            if isinstance(cfg.get("dtype"), dict):
                cfg["dtype"] = "float32"

        # Patch activation configs: TF.js wants {"class_name":"relu"} not {"module":...}
        for key, val in list(obj.items()):
            if key == "activation" and isinstance(val, dict):
                # Ensure simple {class_name, config} form that TF.js understands
                if "class_name" not in val and "module" in val:
                    # Try to extract the activation name from 'module' path
                    mod = val.get("module", "")
                    fn_name = mod.split(".")[-1] if mod else "linear"
                    obj[key] = {"class_name": fn_name, "config": {}}
                elif "class_name" in val and "registered_name" in val:
                    # Keep class_name, drop Keras-3-only fields if they confuse TF.js
                    obj[key] = {"class_name": val["class_name"],
                                "config": val.get("config", {})}
            else:
                obj[key] = patch_for_tfjs(val)
        return obj
    elif isinstance(obj, list):
        return [patch_for_tfjs(item) for item in obj]
    return obj


model_topology = patch_for_tfjs(model_topology)

# TF.js loadLayersModel expects keras_version + backend at topology root
if "keras_version" not in model_topology:
    model_topology["keras_version"] = "2.9.0"
if "backend" not in model_topology:
    model_topology["backend"] = "tensorflow"

model_json = {
    "format":          "layers-model",
    "generatedBy":     "write_tfjs.py (custom)",
    "convertedBy":     "TensorFlow 2.x (manual)",
    "modelTopology":   model_topology,
    "weightsManifest": manifest,
}

with open(os.path.join(OUT_DIR, "model.json"), "w") as f:
    json.dump(model_json, f, indent=2)

print("      Wrote model.json")

# Copy class indices for the web app
if os.path.exists(IDX_PATH):
    shutil.copy(IDX_PATH, "web/class_indices.json")
    print("      Copied class_indices.json -> web/")

print("\n" + "="*55)
print("  CONVERSION COMPLETE")
print(f"  Output: {os.path.abspath(OUT_DIR)}/")
print("  Open web/index.html in a browser to test!")
print("="*55)
