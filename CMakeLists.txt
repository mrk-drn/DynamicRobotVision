cmake_minimum_required(VERSION 3.16)

project(AdaptivePerception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)

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
)

if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found!")
endif()

# Application
add_executable(AdaptivePerception
    src/main.cpp
    src/MotionEstimator.cpp
    src/DepthEstimator.cpp
)

target_include_directories(AdaptivePerception PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
    ${CUDNN_ROOT}/include
)

target_link_libraries(AdaptivePerception PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    ${CUDNN_LIBRARY}
)

# Runtime library paths
set_target_properties(AdaptivePerception PROPERTIES
    BUILD_RPATH
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib;$ORIGIN/../third_party/cudnn/lib;/usr/local/cuda/targets/x86_64-linux/lib"

    BUILD_RPATH_USE_ORIGIN TRUE
)
