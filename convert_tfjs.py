"""
TF.js conversion with lazy submodule stubs for missing optional deps.
Run: python convert_tfjs.py
"""
import sys
import types
import os
import shutil
import subprocess


class LazyStubModule(types.ModuleType):
    """Module stub that auto-creates sub-stub attributes, acting like a package."""
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []   # marks it as a package
        self.__spec__ = None

    def __getattr__(self, item):
        full_name = f"{self.__name__}.{item}"
        if full_name not in sys.modules:
            stub = LazyStubModule(full_name)
            sys.modules[full_name] = stub
        return sys.modules[full_name]

    def __call__(self, *a, **kw):
        return None


# Register top-level stubs for optional heavy deps
for _mod in [
    'tensorflow_decision_forests',
    'jax', 'jaxlib',
    'tensorflow_hub',
    'flax', 'optax', 'orbax',
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = LazyStubModule(_mod)

print("[1/3] Importing TensorFlow...")
import tensorflow as tf
print(f"      TF {tf.__version__} OK")

print("[2/3] Loading model fish_classifier.h5 ...")
model = tf.keras.models.load_model("fish_classifier.h5", compile=False)
print(f"      Input : {model.input_shape}")
print(f"      Output: {model.output_shape}")

print("[3/3] Converting to TensorFlow.js format...")
os.makedirs("web/model", exist_ok=True)

# --- Attempt A: tfjs Python API ---
try:
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, "web/model")
    shutil.copy("class_indices.json", "web/class_indices.json")
    print("      Saved  web/model/model.json  [tfjs API]")
    print("      Copied web/class_indices.json")
    print("\nDONE - open web/index.html in browser to test!")
    sys.exit(0)
except Exception as e:
    print(f"      tfjs API failed: {e}")

# --- Attempt B: export SavedModel, then run CLI ---
print("      Exporting SavedModel for CLI conversion...")
saved_dir = "fish_saved_model"
model.export(saved_dir)
print(f"      SavedModel → {saved_dir}/")

converter_exe = r"C:\Users\Name\AppData\Local\Programs\Python\Python310\Scripts\tensorflowjs_converter.exe"
if os.path.exists(converter_exe):
    print("      Running tensorflowjs_converter CLI...")
    result = subprocess.run(
        [converter_exe,
         "--input_format=tf_saved_model",
         "--output_format=tfjs_graph_model",
         "--signature_name=serving_default",
         saved_dir, "web/model"],
        capture_output=True, text=True
    )
    if result.stdout:
        print("CLI stdout:", result.stdout[:300])
    if result.stderr:
        print("CLI stderr:", result.stderr[:300])
    if result.returncode == 0 and os.path.exists("web/model/model.json"):
        shutil.copy("class_indices.json", "web/class_indices.json")
        print("\nDONE - open web/index.html in browser to test!")
        sys.exit(0)
    else:
        print(f"  CLI converter failed (exit code {result.returncode})")
else:
    print(f"  CLI not found: {converter_exe}")

print("\n[MANUAL STEP REQUIRED]")
print(f"  tensorflowjs_converter --input_format=tf_saved_model {saved_dir} web/model")
sys.exit(1)
