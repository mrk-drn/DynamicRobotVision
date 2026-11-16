import onnxruntime as ort
import numpy as np

model_path = "models/yolo26s-seg.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

print("\nMODEL INPUTS")

for inp in session.get_inputs():
    print(f"Name: {inp.name}")
    print(f"Shape: {inp.shape}")
    print(f"Type: {inp.type}")

print("\nMODEL OUTPUTS")

for out in session.get_outputs():
    print(f"Name: {out.name}")
    print(f"Shape: {out.shape}")
    print(f"Type: {out.type}")

print("\nModel inspection successful!")