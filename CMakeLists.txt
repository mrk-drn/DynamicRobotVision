
cmake_minimum_required(VERSION 3.17)

project(AdaptivePerception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV with CUDA modules
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    video
    videoio
    dnn
    cudaarithm
    cudaimgproc
    cudawarping
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# ONNX Runtime
set(ONNXRUNTIME_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1"
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

# cuDNN
set(CUDNN_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/cudnn"
)

find_library(
    CUDNN_LIBRARY
    NAMES cudnn
    PATHS "${CUDNN_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

# Source files
set(SOURCES
    src/main.cpp

    # Core
    src/core/PredictionManager.cpp

    # Estimators
    src/estimators/MotionEstimator.cpp
    src/estimators/DepthEstimator.cpp
    src/estimators/SegmentationEstimator.cpp

    # Visualization
    src/visualization/Visualizer.cpp
)

# Application
add_executable(AdaptiveRobotVision
    ${SOURCES}
)

# Include directories
target_include_directories(AdaptiveRobotVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
    ${CUDNN_ROOT}/include
)

# Link libraries
target_link_libraries(AdaptiveRobotVision PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    ${CUDNN_LIBRARY}
)

# Compiler options
target_compile_options(AdaptiveRobotVision PRIVATE
    -O3
    -Wall
    -Wextra
)

# Runtime library paths
set_target_properties(AdaptiveRobotVision PROPERTIES
    BUILD_RPATH
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib;$ORIGIN/../third_party/cudnn/lib;/usr/local/cuda/targets/x86_64-linux/lib"

    INSTALL_RPATH
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib;$ORIGIN/../third_party/cudnn/lib;/usr/local/cuda/targets/x86_64-linux/lib"

    BUILD_RPATH_USE_ORIGIN TRUE
)
