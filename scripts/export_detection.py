from ultralytics import YOLO
from pathlib import Path
import shutil

model_path = Path("yolo26s.pt")

model = YOLO(str(model_path))

exported_path = Path(model.export(
    format="onnx",
    imgsz=768,
    simplify=True,
    nms=False
))

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

target_path = models_dir / "yolo26s-detect.onnx"

shutil.move(str(exported_path), str(target_path))

# Delete original PyTorch weights
model_path.unlink(missing_ok=True)

print(f"ONNX model saved to: {target_path.resolve()}")
print("Original .pt model deleted.")