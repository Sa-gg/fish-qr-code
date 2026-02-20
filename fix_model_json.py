"""
fix_model_json.py  --  Patch Keras 3 model.json to be TF.js 4.x compatible.

Keras 3 (shipped with TF 2.16+) serialises model topology in a NEW format
that TensorFlow.js 4.x loadLayersModel() cannot parse.  This script
post-processes the model.json to convert it back to the Keras-2 style
topology that TF.js understands.

Fixes applied:
  1. InputLayer  batch_shape  -> batchInputShape
  2. inbound_nodes  {args,kwargs} objects -> [[layerName, nodeIdx, tensorIdx, kwargs]] arrays
  3. dtype  DTypePolicy objects -> plain "float32" strings
  4. input_layers / output_layers  flat [name,0,0] -> wrapped [[name,0,0]]
  5. Serialised objects (initializers, regularizers, etc.) -> strip module / registered_name
  6. keras_version / backend fields ensured at topology root

Usage:
    py fix_model_json.py
"""

import json, os, sys, copy

MODEL_JSON = os.path.join("web", "model", "model.json")


# ── helpers ────────────────────────────────────────────────────────────────

def simplify_dtype(val):
    """Convert DTypePolicy object to plain string, e.g. 'float32'."""
    if isinstance(val, dict) and val.get("class_name") == "DTypePolicy":
        return val.get("config", {}).get("name", "float32")
    return val


def strip_module(obj):
    """Remove 'module' and 'registered_name' from a serialised Keras object."""
    if isinstance(obj, dict):
        obj.pop("module", None)
        obj.pop("registered_name", None)
    return obj


# Keras-3 uses uppercase class names for regularizers, but TF.js needs lowercase.
# Initializers must stay PascalCase (TF.js expects Zeros, GlorotUniform, etc.)
CLASS_NAME_MAP = {
    # Regularizers -> TF.js only has L1L2 class for l1/l2/l1l2
    "L1": "L1L2", "L2": "L1L2", "L1L2": "L1L2",
    "l1": "L1L2", "l2": "L1L2", "l1l2": "L1L2",
    # Constraints -> lowercase  
    "MaxNorm": "maxNorm", "MinMaxNorm": "minMaxNorm",
    "NonNeg": "nonNeg", "UnitNorm": "unitNorm",
}

# Initializers that TF.js expects in PascalCase (do NOT lowercase these)
VALID_INITIALIZERS = {
    "Zeros", "Ones", "Constant", "RandomUniform", "RandomNormal",
    "TruncatedNormal", "VarianceScaling", "Orthogonal", "Identity",
    "GlorotNormal", "GlorotUniform", "HeNormal", "HeUniform",
    "LeCunNormal", "LeCunUniform", "LecunNormal", "LecunUniform",
}


def normalise_class_name(obj):
    """Normalize class_name for TF.js: regularizers->lowercase, initializers stay PascalCase."""
    if isinstance(obj, dict) and "class_name" in obj:
        cn = obj["class_name"]
        # Only map regularizers/constraints (initializers stay PascalCase)
        if cn in CLASS_NAME_MAP:
            obj["class_name"] = CLASS_NAME_MAP[cn]


def extract_keras_history(tensor_obj):
    """
    Given a Keras-3 __keras_tensor__ object, return [layerName, nodeIdx, tensorIdx].
    """
    if isinstance(tensor_obj, dict) and tensor_obj.get("class_name") == "__keras_tensor__":
        cfg = tensor_obj.get("config", {})
        history = cfg.get("keras_history", ["", 0, 0])
        return list(history)          # [layerName, nodeIndex, tensorIndex]
    return None


def convert_inbound_node(node_obj):
    """
    Convert ONE Keras-3 inbound_node entry (a dict with 'args'+'kwargs')
    into the Keras-2 array format that TF.js expects.

    Keras-3 single input:
      {"args": [<tensor>], "kwargs": {..}}          -> [[layerName, nIdx, tIdx, kwargs]]

    Keras-3 list input (Add / Concatenate):
      {"args": [[<tensor>, <tensor>]], "kwargs": {}} -> [[l1,n,t,kw], [l2,n,t,kw]]
    """
    if not isinstance(node_obj, dict) or "args" not in node_obj:
        # Already array or unknown format; leave as-is
        return node_obj

    args   = node_obj.get("args", [])
    kwargs = node_obj.get("kwargs", {})

    # Clean kwargs: remove None values and 'mask' which TF.js ignores
    clean_kw = {}
    for k, v in kwargs.items():
        if v is not None and k != "mask":
            clean_kw[k] = v

    result = []

    if len(args) == 1:
        item = args[0]
        if isinstance(item, list):
            # Multiple tensors passed as a list (e.g. Add layer)
            for tensor in item:
                hist = extract_keras_history(tensor)
                if hist:
                    result.append(hist + [clean_kw])
                else:
                    result.append(tensor)
        else:
            hist = extract_keras_history(item)
            if hist:
                result.append(hist + [clean_kw])
            else:
                result.append(item)
    else:
        # Multiple positional args (rare, but handle)
        for item in args:
            hist = extract_keras_history(item)
            if hist:
                result.append(hist + [clean_kw])
            else:
                result.append(item)

    return result


def fix_layer(layer):
    """Recursively fix a single layer dict."""
    cfg = layer.get("config", {})

    # ── 1. InputLayer: batch_shape -> batchInputShape ──────────────────
    if layer.get("class_name") == "InputLayer":
        if "batch_shape" in cfg and "batchInputShape" not in cfg:
            cfg["batchInputShape"] = cfg.pop("batch_shape")
        # Also ensure batch_input_shape alias
        if "batchInputShape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg["batchInputShape"]

    # ── 2. dtype: DTypePolicy -> simple string ─────────────────────────
    if "dtype" in cfg:
        cfg["dtype"] = simplify_dtype(cfg["dtype"])

    # ── 3. Strip module/registered_name & normalise class names ────────
    for key in list(cfg.keys()):
        val = cfg[key]
        if isinstance(val, dict) and "class_name" in val:
            strip_module(val)
            normalise_class_name(val)
            # Also recurse into config of sub-objects
            sub_cfg = val.get("config", {})
            if isinstance(sub_cfg, dict):
                for sk in list(sub_cfg.keys()):
                    if isinstance(sub_cfg[sk], dict) and "class_name" in sub_cfg[sk]:
                        strip_module(sub_cfg[sk])
                        normalise_class_name(sub_cfg[sk])

    # ── 4. inbound_nodes: convert from Keras-3 to Keras-2 format ──────
    nodes = layer.get("inbound_nodes", [])
    new_nodes = []
    for node in nodes:
        if isinstance(node, dict) and "args" in node:
            converted = convert_inbound_node(node)
            new_nodes.append(converted)
        else:
            # Already in array form
            new_nodes.append(node)
    layer["inbound_nodes"] = new_nodes

    # ── 5. Nested Functional model (e.g. MobileNetV2 sub-model) ────────
    if layer.get("class_name") == "Functional" or layer.get("class_name") == "Sequential":
        fix_functional_config(cfg)


def fix_io_layers(cfg, key):
    """
    Fix input_layers / output_layers.
    Keras-3: ["name", 0, 0]       (flat list)
    TF.js:   [["name", 0, 0]]     (wrapped in outer list)
    """
    val = cfg.get(key)
    if val is None:
        return
    # If it's a flat list like ["name", 0, 0]
    if isinstance(val, list) and len(val) == 3 and isinstance(val[0], str):
        cfg[key] = [val]
    # If it's already wrapped but each entry is flat
    elif isinstance(val, list) and len(val) > 0:
        if isinstance(val[0], str):
            cfg[key] = [val]


def flatten_nested_model(layers, outer_input_name):
    """
    Flatten nested Functional models (e.g., MobileNetV2) into the top-level layer list.
    This fixes 'Graph disconnected' errors in TF.js.
    
    Returns a new list of layers with nested models expanded inline.
    """
    new_layers = []
    nested_output_map = {}  # Maps nested model name -> its actual output layer name
    
    for layer in layers:
        class_name = layer.get("class_name")
        layer_name = layer.get("name")
        
        if class_name == "Functional":
            # This is a nested model - flatten it
            inner_cfg = layer.get("config", {})
            inner_layers = inner_cfg.get("layers", [])
            inner_input_layers = inner_cfg.get("input_layers", [])
            inner_output_layers = inner_cfg.get("output_layers", [])
            
            # Get the outer input that feeds into this nested model
            inbound = layer.get("inbound_nodes", [])
            if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
                # inbound_nodes[0][0] = [layer_name, node_idx, tensor_idx, kwargs]
                outer_input = inbound[0][0][0] if isinstance(inbound[0][0], list) else outer_input_name
            else:
                outer_input = outer_input_name
            
            # Find the inner InputLayer name
            inner_input_name = None
            for il in inner_layers:
                if il.get("class_name") == "InputLayer":
                    inner_input_name = il.get("name")
                    break
            
            # Find the inner output layer name
            inner_output_name = None
            if inner_output_layers and len(inner_output_layers) > 0:
                inner_output_name = inner_output_layers[0][0] if isinstance(inner_output_layers[0], list) else inner_output_layers[0]
            
            # Map this nested model's name to its actual output
            nested_output_map[layer_name] = inner_output_name
            
            # Add all inner layers (except InputLayer), remapping input references
            for il in inner_layers:
                if il.get("class_name") == "InputLayer":
                    # Skip inner InputLayer - we use outer input instead
                    continue
                
                # Remap inbound_nodes: replace inner_input_name with outer_input
                il_inbound = il.get("inbound_nodes", [])
                for node in il_inbound:
                    if isinstance(node, list):
                        for conn in node:
                            if isinstance(conn, list) and len(conn) > 0:
                                if conn[0] == inner_input_name:
                                    conn[0] = outer_input
                
                new_layers.append(il)
        else:
            # Regular layer - but remap references to nested models to their actual outputs
            inbound = layer.get("inbound_nodes", [])
            for node in inbound:
                if isinstance(node, list):
                    for conn in node:
                        if isinstance(conn, list) and len(conn) > 0:
                            if conn[0] in nested_output_map:
                                conn[0] = nested_output_map[conn[0]]
            
            new_layers.append(layer)
    
    return new_layers


def fix_functional_config(cfg):
    """Fix a Functional model config (top-level or nested)."""
    # Fix layers inside this Functional model
    layers = cfg.get("layers", [])
    for layer in layers:
        fix_layer(layer)

    # Flatten nested Functional models
    input_layers = cfg.get("input_layers", [])
    outer_input_name = None
    if input_layers:
        if isinstance(input_layers[0], list):
            outer_input_name = input_layers[0][0]
        else:
            outer_input_name = input_layers[0]
    
    # Check if there are nested Functional models
    has_nested = any(l.get("class_name") == "Functional" for l in layers)
    if has_nested and outer_input_name:
        cfg["layers"] = flatten_nested_model(layers, outer_input_name)

    # Fix input_layers / output_layers wrapping
    fix_io_layers(cfg, "input_layers")
    fix_io_layers(cfg, "output_layers")


def fix_topology(topology):
    """Fix the entire modelTopology dict."""

    # Ensure keras_version and backend at root
    if "keras_version" not in topology:
        topology["keras_version"] = "2.15.0"
    if "backend" not in topology:
        topology["backend"] = "tensorflow"

    model_config = topology.get("model_config", {})
    strip_module(model_config)
    cfg = model_config.get("config", {})

    if model_config.get("class_name") in ("Functional", "Sequential"):
        fix_functional_config(cfg)

    # Also fix training_config if present (strip module from optimizer, etc.)
    training = topology.get("training_config", {})
    opt_cfg = training.get("optimizer_config", {})
    strip_module(opt_cfg)


def fix_weights_manifest(data):
    """
    Fix weight names in weightsManifest to match TF.js expectations.
    
    For DepthwiseConv2D layers:
      Keras saves: layer_name/kernel
      TF.js expects: layer_name/depthwise_kernel
    """
    # First, collect all DepthwiseConv2D layer names
    depthwise_layers = set()
    cfg = data.get("modelTopology", {}).get("model_config", {}).get("config", {})
    for layer in cfg.get("layers", []):
        if layer.get("class_name") == "DepthwiseConv2D":
            depthwise_layers.add(layer.get("name"))
    
    # Fix weight names
    for manifest in data.get("weightsManifest", []):
        for weight in manifest.get("weights", []):
            name = weight.get("name", "")
            parts = name.split("/")
            if len(parts) == 2:
                layer_name, weight_type = parts
                # For DepthwiseConv2D, rename kernel -> depthwise_kernel
                if layer_name in depthwise_layers and weight_type == "kernel":
                    weight["name"] = f"{layer_name}/depthwise_kernel"
                    print(f"  Renamed: {name} -> {weight['name']}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_JSON):
        print(f"[ERROR] {MODEL_JSON} not found.")
        sys.exit(1)

    print(f"Reading {MODEL_JSON} ...")
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    topology = data.get("modelTopology", {})
    fix_topology(topology)
    
    # Fix weight names in weightsManifest
    fix_weights_manifest(data)

    # Write back
    with open(MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f)

    size_kb = os.path.getsize(MODEL_JSON) / 1024
    print(f"Fixed model.json written ({size_kb:.0f} KB)")
    print("Done -- refresh browser to test.")


if __name__ == "__main__":
    main()
