from ultralytics import YOLO
from pathlib import Path
import shutil

model_path = Path("yolo26n-depth.pt")

model = YOLO(str(model_path))

exported_path = Path(model.export(
    format="onnx",
    imgsz=768,
    simplify=True
))

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

target_path = models_dir / "yolo26n-depth.onnx"

shutil.move(str(exported_path), str(target_path))

# Delete original PyTorch weights
model_path.unlink(missing_ok=True)

print(f"ONNX model saved to: {target_path.resolve()}")
print("Original .pt model deleted.")