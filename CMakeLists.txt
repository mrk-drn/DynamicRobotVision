cmake_minimum_required(VERSION 3.17)

project(DynamicRobotVision LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

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
    cudafeatures2d
    cudaoptflow
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    FiltersCore
    RenderingCore
    RenderingOpenGL2
)

message(STATUS "VTK version: ${VTK_VERSION}")

# ONNX Runtime configuration

set(ONNXRUNTIME_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1"
    CACHE PATH "Path to ONNX Runtime installation"
)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)

set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# cuDNN configuration

set(CUDNN_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/cudnn"
    CACHE PATH "Path to cuDNN installation"
)

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS "${CUDNN_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    PATHS "${CUDNN_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

# Source files

set(SOURCES
    src/main.cpp

    src/core/PredictionManager.cpp

    src/estimators/MotionEstimator.cpp
    src/estimators/MotionEstimatorKernels.cu

    src/estimators/DepthEstimator.cpp
    src/estimators/DepthKernels.cu

    src/estimators/SegmentationEstimator.cpp
    src/estimators/SegmentationKernels.cu

    src/estimators/ObjectDetectionEstimator.cpp
    src/estimators/ObjectDetectionKernels.cu

    src/visualization/Visualizer.cpp
    src/visualization/PointCloudVisualizer.cpp
    src/visualization/PointCloudKernels.cu
)

# Executable

add_executable(DynamicRobotVision ${SOURCES})

# Language standards

set_target_properties(DynamicRobotVision PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON

    CUDA_SEPARABLE_COMPILATION OFF
)

# Include directories

target_include_directories(DynamicRobotVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

# Link libraries

target_link_libraries(DynamicRobotVision PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}

    ONNXRuntime::ONNXRuntime
    ${CUDNN_LIBRARY}

    CUDA::cudart

    OpenGL::GL
    glfw
    GLEW::GLEW

    ${VTK_LIBRARIES}
)

# Compiler options

target_compile_options(DynamicRobotVision PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-fno-omit-frame-pointer>
)

# Runtime library paths

set(RUNTIME_LIBRARY_PATHS
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib"
    "$ORIGIN/../third_party/cudnn/lib"
    "${CUDAToolkit_LIBRARY_DIR}"
)

set_target_properties(DynamicRobotVision PROPERTIES
    BUILD_RPATH "${RUNTIME_LIBRARY_PATHS}"
    INSTALL_RPATH "${RUNTIME_LIBRARY_PATHS}"
    BUILD_RPATH_USE_ORIGIN TRUE
)

# VTK module initialization

vtk_module_autoinit(
    TARGETS DynamicRobotVision
    MODULES ${VTK_LIBRARIES}
)
