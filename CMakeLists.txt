cmake_minimum_required(VERSION 3.17)

project(AdaptivePerception LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

set(ONNXRUNTIME_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1"
)

find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
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

set(CUDNN_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/cudnn"
)

find_path(CUDNN_INCLUDE_DIR
    cudnn.h
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

add_executable(AdaptiveRobotVision ${SOURCES})

target_include_directories(AdaptiveRobotVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

target_link_libraries(AdaptiveRobotVision PRIVATE
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

target_compile_options(AdaptiveRobotVision PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
)

set_target_properties(AdaptiveRobotVision PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF

    BUILD_RPATH
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib;$ORIGIN/../third_party/cudnn/lib;/usr/local/cuda/targets/x86_64-linux/lib"

    INSTALL_RPATH
    "$ORIGIN/../third_party/onnx/onnxruntime-linux-x64-gpu_cuda13-1.30.1/lib;$ORIGIN/../third_party/cudnn/lib;/usr/local/cuda/targets/x86_64-linux/lib"

    BUILD_RPATH_USE_ORIGIN TRUE
)

vtk_module_autoinit(
    TARGETS AdaptiveRobotVision
    MODULES ${VTK_LIBRARIES}
)