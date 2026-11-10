cmake_minimum_required(VERSION 3.16)

project(AdaptivePerception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-gpu_cuda12-1.30.1"
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found!")
endif()

add_executable(AdaptivePerception
    src/main.cpp
    src/MotionEstimator.cpp
    src/DepthEstimator.cpp
)

target_include_directories(AdaptivePerception PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(AdaptivePerception PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

set_target_properties(AdaptivePerception PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
