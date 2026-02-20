# Summary Report: Fixing TF.js Model Loading Errors
**Philippine Fish Classifier -- Keras 3 to TF.js 4.x Compatibility**

---

## Problem

After training the MobileNetV2 fish classifier and converting it to TensorFlow.js
format, the browser web app failed to load the model with **two errors**:

1. **`Corrupted configuration, expected array for nodeData: [object Object]`**
   - Source: TF.js `container.js:1276`
2. **`An InputLayer should be passed either a batchInputShape or an inputShape`**
   - Source: TF.js `tf.min.js`

The model was trained using **TensorFlow 2.20.0 + Keras 3.12.1** and converted
with **tensorflowjs 4.22.0**. The browser loads it with **TF.js 4.17.0**.

---

## Root Cause

**Keras 3 changed the model serialization format** (the JSON topology), but
TF.js 4.17 still expects the old Keras-2 format. The tensorflowjs converter
(v4.22.0) does NOT translate the topology -- it passes Keras 3's JSON through
as-is, creating an incompatibility.

### Specific Incompatibilities Found

| Field | Keras 3 Format | TF.js Expected Format |
|-------|----------------|----------------------|
| `InputLayer.batch_shape` | `"batch_shape": [null,224,224,3]` | `"batchInputShape": [null,224,224,3]` |
| `inbound_nodes` | `[{"args": [<tensor_obj>], "kwargs": {...}}]` | `[[["layerName", 0, 0, {}]]]` |
| `dtype` | `{"class_name":"DTypePolicy","config":{"name":"float32"}}` | `"float32"` |
| `input_layers` / `output_layers` | `["name", 0, 0]` (flat) | `[["name", 0, 0]]` (wrapped) |
| Initializers, Regularizers | Include `module` and `registered_name` keys | Only `class_name` + `config` |

The **nodeData error** was caused by `inbound_nodes` being objects (`{args, kwargs}`)
when TF.js expects nested arrays. This is the most critical breaking change
between Keras 2 and Keras 3 serialization.

---

## Solution

Created **`fix_model_json.py`** -- a post-processing script that transforms the
Keras-3 model.json into the Keras-2 format TF.js expects.

### Workflow
```
train_fish_classifier.py       (trains model, saves .h5)
        |
        v
convert_model.py               (converts .h5 -> TFLite + TF.js)
        |
        v
fix_model_json.py              (patches model.json for TF.js compat)
        |
        v
web/model/model.json           (ready for browser!)
```

### Transformations Applied by fix_model_json.py

1. **InputLayer config**: Renamed `batch_shape` to `batchInputShape` (and added
   `batch_input_shape` alias). Applied to ALL InputLayers including nested ones.

2. **inbound_nodes**: Converted Keras-3 object format to Keras-2 array format:
   - Single input: `{"args":[<tensor>],"kwargs":{}}` --> `[["layerName",0,0,{}]]`
   - Multi input (Add/Concat): `{"args":[[<t1>,<t2>]],"kwargs":{}}` -->
     `[["layer1",0,0,{}],["layer2",0,0,{}]]`
   - Extracts `keras_history` from `__keras_tensor__` objects to get `[name, nodeIdx, tensorIdx]`

3. **dtype**: Replaced DTypePolicy objects with plain string `"float32"`.

4. **input_layers / output_layers**: Wrapped flat `["name",0,0]` in extra array
   to get `[["name",0,0]]` as TF.js expects.

5. **Stripped `module` and `registered_name`** from ALL serialized sub-objects
   (initializers, regularizers, optimizers).

### Result
- model.json size: 158 KB --> 103 KB (removed redundant Keras-3 metadata)
- All 161 layers (154 inner MobileNetV2 + 7 outer) successfully patched
- Zero dict-type inbound_nodes remaining

---

## Files Modified / Created

| File | Action | Purpose |
|------|--------|---------|
| `fix_model_json.py` | **Created** | Post-processes model.json for TF.js compat |
| `web/model/model.json` | **Patched** | Fixed Keras-3 topology to Keras-2 format |

---

## How to Reproduce (for future retraining)

If you retrain the model or re-run conversion, always run the fix script after:

```bash
py train_fish_classifier.py      # Train model
py convert_model.py              # Convert to TFLite + TF.js
py fix_model_json.py             # Fix model.json for browser
```

Then serve:
```bash
cd web
py -m http.server 8080
```

---

## Technical Details

- **Python**: 3.10.11
- **TensorFlow**: 2.20.0 (CPU, Keras 3.12.1)
- **tensorflowjs**: 4.22.0 (installed with --no-deps)
- **TF.js (browser)**: 4.17.0 via CDN
- **Model**: MobileNetV2, 2.6M params, input (224,224,3), 3 classes
- **Classes**: Bangus, Galunggong, Tilapia
- **Training accuracy**: ~46.7% validation (limited 57-image dataset)
