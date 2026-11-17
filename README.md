DynamicRobotVision

Resource-Aware Visual Perception and Qualitative 3D Reconstruction for Robotics and Autonomous Systems

DynamicRobotVision is an experimental computer vision project exploring dynamic inference scheduling under GPU resource constraints. It combines depth estimation, object detection, and semantic segmentation with CUDA-accelerated processing to investigate how multiple perception capabilities can be coordinated without executing every model continuously.

The project explores a simple idea: instead of running every perception model at all times, can we allocate computational resources dynamically according to the current scene and perception requirements?

Demonstrations

The following videos demonstrate the perception pipeline and its qualitative 3D reconstruction capabilities.

<!-- Demo 1: Replace with your first demonstration video -->

Demo 1 – Dynamic Visual Perception

<video src="assets/demo-1.mp4" controls width="100%"></video>

<!-- Demo 2: Replace with your second demonstration video -->

Demo 2 – Qualitative 3D Reconstruction

<video src="assets/demo-2.mp4" controls width="100%"></video>

Replace the video paths with your actual video files or GitHub-hosted video URLs. The videos are intentionally arranged vertically to accommodate their horizontal aspect ratio.

Motivation

Robotic systems and autonomous vehicles often operate under strict computational and power constraints. While modern deep learning models provide increasingly sophisticated perception capabilities, running multiple neural networks simultaneously can be expensive, particularly on embedded or resource-constrained GPU platforms.

This project explores an alternative to continuously executing all available perception models.

Rather than treating depth estimation, object detection, and segmentation as independent processes that must always run in parallel, DynamicRobotVision uses a lightweight motion-based scheduling heuristic to alternate between different perception tasks.

The underlying motivation is that perception requirements can vary with the environment:

Moving environments: More precise object boundaries can be particularly useful when reasoning about obstacles and potential collisions. Therefore, semantic segmentation is preferred when motion is detected.
Relatively static environments: Object detection provides a more lightweight representation of the scene and is selected when significant motion is not detected.
Depth estimation: Runs independently of the motion-based decision and provides depth information that can be combined with the latest available perception results.

The objective is to explore whether dynamically allocating inference resources can provide useful visual understanding without requiring all perception models to execute continuously.

This is an experimental exploration rather than an attempt to establish a new perception paradigm or claim state-of-the-art performance.

System Overview

The current implementation combines three neural network-based perception tasks with GPU-accelerated motion estimation and point-cloud reconstruction.

Perception Pipeline
Component	Function
Depth Estimation	Predicts per-pixel depth information.
Object Detection	Identifies objects and provides detection-based visual representations.
Semantic Segmentation	Produces pixel-level semantic representations with more detailed object boundaries.
Motion Estimation	Classifies the current scene as moving or relatively static.
3D Reconstruction	Combines the latest depth prediction and perception colour map into a qualitative 3D point-cloud visualization.
Dynamic Inference Scheduling

The current scheduler operates at a target prediction frequency of 15 FPS and alternates between depth estimation and a motion-dependent perception task.

The scheduling logic is:

Depth estimation: Every other scheduled inference, the depth estimation model is executed.
Motion-dependent perception: On alternating inference steps:
If motion is detected, semantic segmentation is executed.
Otherwise, object detection is executed.
3D reconstruction: The latest depth prediction is combined with the most recently available segmentation or detection colour map.

Importantly, depth estimation does not overwrite the previously available perception colour map. This allows the reconstruction to retain the latest segmentation or detection information while new depth predictions are generated.

The current motion-based scheduling policy is a heuristic motivated by the assumption that moving environments require more precise object boundaries for potentially collision-relevant perception, whereas object detection can be sufficient in relatively static situations.

Qualitative 3D Reconstruction

The project also includes a CUDA-accelerated point-cloud visualization pipeline.

It combines:

Predicted depth maps.
Pixel-level colour information from the latest perception result.
Camera intrinsic parameters.
A configurable sampling strategy.

The resulting visualization provides a qualitative proof of concept for maintaining a form of 3D scene understanding using dynamically scheduled perception tasks.

It is intended as an experimental visualization rather than a metrically validated 3D reconstruction system.

Technical Implementation

The project uses a combination of C++, CUDA, OpenCV, and ONNX Runtime.

Core technologies

C++20 – Main application and perception pipeline.
CUDA – Custom GPU kernels and accelerated processing.
OpenCV 4.14.0 – Image processing, video input/output, visualization, and CUDA-accelerated motion estimation.
ONNX Runtime 1.30.1 (CUDA 13) – GPU-based neural network inference.
cuDNN 9.28.1 – GPU-accelerated deep learning operations.
Eigen 3.4 – Linear algebra support.
GPU Acceleration

Several components use GPU acceleration:

Motion estimation using CUDA-based feature detection and sparse optical flow.
Custom CUDA kernels for motion displacement reduction.
GPU-based ONNX inference.
Custom CUDA kernels for point-cloud rendering.

The point-cloud renderer uses GPU-side projection, depth ordering, and framebuffer operations before transferring the resulting visualization back to the CPU.

Persistent GPU buffers are used in the custom rendering implementation to reduce repeated memory allocations.

Neural Network Models

The current implementation uses the following Ultralytics YOLO26 models exported to ONNX:

Task	Model
Depth Estimation	YOLO26s-Depth
Semantic Segmentation	YOLO26s-Seg
Object Detection	YOLO26s

The models are used in their original pretrained form. No task-specific fine-tuning was performed as part of this project.

ONNX export scripts are provided in the scripts/ directory.

Hardware

Development hardware

GPU: NVIDIA GeForce RTX 4050 Laptop GPU
GPU memory: 6 GB
CUDA driver interface: CUDA 13.3

The project was developed and experimented with on a relatively resource-constrained laptop GPU.

This hardware constraint is an important part of the motivation: the objective is to explore how perception workloads can be coordinated when computational resources are limited, rather than assuming access to high-end GPU hardware.

Requirements

The following software components are required:

Linux (development environment: Ubuntu-based).
CMake 3.17 or newer.
C++20-compatible compiler.
NVIDIA CUDA Toolkit compatible with CUDA 13.
CUDA-enabled OpenCV 4.14.0.
ONNX Runtime GPU 1.30.1 built for CUDA 13.
cuDNN 9.
Eigen 3.4.

The project also expects the ONNX Runtime and cuDNN development files to be available in the existing third_party/ directory.

Note: OpenCV must include the CUDA modules used by the project, particularly cudaarithm, cudaimgproc, cudawarping, cudafeatures2d, and cudaoptflow.

Building

After installing the required dependencies and placing the ONNX Runtime and cuDNN libraries in their expected locations, build the project from its root directory.

bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)


The build configuration expects the following dependency locations:

text
third_party/
├── onnx/
│   └── onnxruntime-linux-x64-gpu_cuda13-1.30.1/
│       ├── include/
│       └── lib/
│
└── cudnn/
    ├── include/
    └── lib/


The CMake configuration locates these dependencies relative to the project directory.

Running

The application expects the exported ONNX models in the models/ directory:

text
models/
├── yolo26s-depth.onnx
├── yolo26s-seg.onnx
└── yolo26s-detect.onnx


Set the runtime library search path and launch the application:

bash
export LD_LIBRARY_PATH="$PWD/third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib:$PWD/third_party/cudnn/lib:/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

./build/DynamicRobotVision


By default, the application attempts to open the first available camera.

Alternatively, provide a video file:

bash
./build/DynamicRobotVision path/to/video.mp4

Project Structure
text
DynamicRobotVision/
│
├── include/
│   ├── core/
│   ├── estimators/
│   └── visualization/
│
├── src/
│   ├── core/
│   ├── estimators/
│   └── visualization/
│
├── scripts/
│   ├── export_depth.py
│   ├── export_segmentation.py
│   └── export_detection.py
│
├── models/
│
├── assets/
│   └── videos/
│
├── third_party/
│   ├── onnx/
│   └── cudnn/
│
├── CMakeLists.txt
└── README.md

Future Work

Several directions could extend the current implementation:

1. Learned Perception Scheduling

Replace the current motion-based heuristic with a learned scheduling policy that determines which perception tasks should be executed based on scene characteristics, temporal context, computational availability, and downstream requirements.

2. More Capable Perception Models

Investigate stronger or more efficient depth estimation, detection, and segmentation architectures, including models specifically designed for real-time inference on embedded hardware.

3. Resource-Aware Downstream Systems

Explore architectures in which downstream tasks dynamically request or utilize different perception outputs according to their current requirements.

Potential applications include:

End-to-end autonomous driving systems.
Robot navigation and obstacle avoidance.
Robotic manipulation.
Resource-constrained embodied AI systems.

An interesting longer-term direction would be to investigate how dynamic perception provisioning could be integrated directly into downstream models, allowing computational resources to be allocated according to the requirements of the task rather than relying on a fixed perception pipeline.

Scope and Limitations

The current implementation is an experimental prototype intended to explore resource-aware visual perception.

In particular:

The scheduling policy is heuristic-based and has not been learned or optimized.
Motion is used as a proxy for changing perception requirements.
The neural networks are pretrained models and have not been fine-tuned for a particular robotics or autonomous driving dataset.
The 3D reconstruction is qualitative and has not been evaluated against ground-truth geometry.
No systematic benchmarking of inference latency, energy consumption, or scheduling efficiency has been conducted.

These limitations also define several opportunities for further investigation.

Acknowledgements

The project builds upon pretrained models from the Ultralytics ecosystem, ONNX Runtime, OpenCV, NVIDIA CUDA, and cuDNN.