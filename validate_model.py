"""Validate model.json for TF.js compatibility."""
import json, sys

d = json.load(open("web/model/model.json"))
errors = []

def check_layer(layer, path=""):
    name = layer.get("name", "?")
    full = f"{path}/{name}"
    cfg = layer.get("config", {})

    # Check InputLayer
    if layer.get("class_name") == "InputLayer":
        if "batchInputShape" not in cfg and "batch_input_shape" not in cfg:
            errors.append(f"{full}: Missing batchInputShape (has: {list(cfg.keys())})")

    # Check inbound_nodes are arrays, not dicts
    for i, node in enumerate(layer.get("inbound_nodes", [])):
        if isinstance(node, dict):
            errors.append(f"{full}: inbound_nodes[{i}] is dict, should be array")

    # Check dtype is string not dict
    dt = cfg.get("dtype")
    if isinstance(dt, dict):
        errors.append(f"{full}: dtype is DTypePolicy dict, should be string")

    # Check regularizers have L1L2 class_name (TF.js expects L1L2 for l1/l2)
    UPPER_REGULARIZERS = {"L1", "L2", "l1", "l2"}
    for key in ["kernel_regularizer", "bias_regularizer", "activity_regularizer"]:
        val = cfg.get(key)
        if isinstance(val, dict) and val.get("class_name") in UPPER_REGULARIZERS:
            errors.append(f"{full}: {key}.class_name = '{val['class_name']}' (should be L1L2)")
        if isinstance(val, dict) and "module" in val:
            errors.append(f"{full}: {key} still has 'module' field")

    # Check initializers have PascalCase class_name (TF.js expects Zeros, GlorotUniform, etc.)
    VALID_INIT_CLASSES = {"Zeros", "Ones", "Constant", "RandomUniform", "RandomNormal",
                          "TruncatedNormal", "VarianceScaling", "Orthogonal", "Identity",
                          "GlorotNormal", "GlorotUniform", "HeNormal", "HeUniform",
                          "LeCunNormal", "LeCunUniform", "LecunNormal", "LecunUniform"}
    for key in ["kernel_initializer", "bias_initializer", "depthwise_initializer", "pointwise_initializer"]:
        val = cfg.get(key)
        if isinstance(val, dict):
            cn = val.get("class_name")
            # Lowercase initializers are WRONG (TF.js expects PascalCase)
            if cn and cn[0].islower() and cn not in VALID_INIT_CLASSES:
                errors.append(f"{full}: {key}.class_name = '{cn}' (should be PascalCase like 'Zeros')")
            if "module" in val:
                errors.append(f"{full}: {key} still has 'module' field")

    # Check nested Functional model
    if layer.get("class_name") in ("Functional", "Sequential"):
        sub_cfg = cfg
        sub_il = sub_cfg.get("input_layers")
        if isinstance(sub_il, list) and len(sub_il) == 3 and isinstance(sub_il[0], str):
            errors.append(f"{full}: input_layers is flat, should be wrapped")
        sub_ol = sub_cfg.get("output_layers")
        if isinstance(sub_ol, list) and len(sub_ol) == 3 and isinstance(sub_ol[0], str):
            errors.append(f"{full}: output_layers is flat, should be wrapped")
        for sub_layer in sub_cfg.get("layers", []):
            check_layer(sub_layer, full)

mc = d["modelTopology"]["model_config"]
top_cfg = mc["config"]
# Check top-level input/output layers
il = top_cfg.get("input_layers")
if isinstance(il, list) and len(il) == 3 and isinstance(il[0], str):
    errors.append("Top: input_layers is flat")
ol = top_cfg.get("output_layers")
if isinstance(ol, list) and len(ol) == 3 and isinstance(ol[0], str):
    errors.append("Top: output_layers is flat")

for layer in top_cfg.get("layers", []):
    check_layer(layer, "top")

if errors:
    print(f"FAILED: {len(errors)} issues found:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("PASSED: model.json is TF.js compatible")
    print(f"  Format: {d.get('format')}")
    print(f"  Layers: {len(top_cfg.get('layers', []))} outer")
    inner = top_cfg.get("layers", [{}])[1] if len(top_cfg.get("layers",[])) > 1 else {}
    if inner.get("class_name") == "Functional":
        print(f"  Inner model layers: {len(inner.get('config',{}).get('layers',[]))}")
    wm = d.get("weightsManifest", [{}])[0]
    print(f"  Weight shards: {len(wm.get('paths',[]))}")
    print(f"  Weight entries: {len(wm.get('weights',[]))}")
