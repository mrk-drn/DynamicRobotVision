
#include "estimators/ObjectDetectionEstimator.hpp"
#include "estimators/ObjectDetectionKernels.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int INPUT_WIDTH = 768;
constexpr int INPUT_HEIGHT = 768;

constexpr int NUM_CLASSES = 80;
constexpr int NUM_DETECTIONS = 300;
constexpr int NUM_VALUES = 6;

constexpr float CONFIDENCE_THRESHOLD = 0.20f;

constexpr std::size_t INPUT_ELEMENTS =
    3ULL * INPUT_WIDTH * INPUT_HEIGHT;

constexpr std::size_t OUTPUT_ELEMENTS =
    NUM_DETECTIONS * NUM_VALUES;

const std::array<std::string, NUM_CLASSES> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard",
    "surfboard", "tennis racket",
    "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant",
    "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink",
    "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// IMPORTANT:
// These are exactly the same colours used for 2D bounding boxes.
const std::array<cv::Scalar, NUM_CLASSES>& getClassColors()
{
    static const std::array<cv::Scalar, NUM_CLASSES> colors = [] {

        std::array<cv::Scalar, NUM_CLASSES> result;

        for (int i = 0; i < NUM_CLASSES; ++i) {

            cv::Mat hsv(
                1, 1,
                CV_8UC3,
                cv::Scalar(
                    (i * 137) % 180,
                    200,
                    230
                )
            );

            cv::Mat bgr;

            cv::cvtColor(
                hsv,
                bgr,
                cv::COLOR_HSV2BGR
            );

            const cv::Vec3b color =
                bgr.at<cv::Vec3b>(0, 0);

            result[i] = cv::Scalar(
                color[0],
                color[1],
                color[2]
            );
        }

        return result;

    }();

    return colors;
}

void checkCuda(
    cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {

        throw std::runtime_error(
            std::string(operation) +
            ": " +
            cudaGetErrorString(status)
        );
    }
}

struct Detection {

    cv::Rect box;

    int classId = -1;

    float confidence = 0.0f;
};

}

ObjectDetectionEstimator::ObjectDetectionEstimator(
    const std::string& modelPath
)
    : env(
        ORT_LOGGING_LEVEL_WARNING,
        "AdaptivePerception"
    ),
      session(nullptr)
{
    std::cout
        << "Loading optimized object detection model: "
        << modelPath
        << std::endl;

    const auto providers =
        Ort::GetAvailableProviders();

    const bool cudaAvailable =
        std::find(
            providers.begin(),
            providers.end(),
            "CUDAExecutionProvider"
        ) != providers.end();

    if (!cudaAvailable) {

        throw std::runtime_error(
            "ONNX Runtime CUDAExecutionProvider unavailable."
        );
    }

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    sessionOptions.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);

    OrtCUDAProviderOptions cudaOptions{};

    cudaOptions.device_id = 0;

    sessionOptions.AppendExecutionProvider_CUDA(
        cudaOptions
    );

    session = Ort::Session(
        env,
        modelPath.c_str(),
        sessionOptions
    );

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session.GetInputNameAllocated(
            0,
            allocator
        );

    auto outputName =
        session.GetOutputNameAllocated(
            0,
            allocator
        );

    inputName_ = inputName.get();
    outputName_ = outputName.get();

    checkCuda(
        cudaSetDevice(0),
        "Selecting CUDA device"
    );

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&gpuInput_),
            INPUT_ELEMENTS * sizeof(float)
        ),
        "Allocating persistent CUDA input"
    );

    std::cout
        << "Object detection model loaded."
        << std::endl;

    std::cout
        << "ONNX Runtime CUDA I/O Binding enabled."
        << std::endl;

    std::cout
        << "Custom CUDA preprocessing enabled."
        << std::endl;
}

ObjectDetectionEstimator::~ObjectDetectionEstimator()
{
    releaseGpuBuffers();
}

void ObjectDetectionEstimator::ensureGpuBuffers(
    int width,
    int height
) {
    const std::size_t required =
        static_cast<std::size_t>(width) *
        height * 3;

    if (required <= gpuFrameCapacity_ && gpuFrame_) {
        return;
    }

    if (gpuFrame_) {

        checkCuda(
            cudaFree(gpuFrame_),
            "Freeing previous frame buffer"
        );

        gpuFrame_ = nullptr;
    }

    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&gpuFrame_),
            required
        ),
        "Allocating CUDA frame buffer"
    );

    gpuFrameCapacity_ = required;
}

void ObjectDetectionEstimator::releaseGpuBuffers() noexcept
{
    if (gpuFrame_) {
        cudaFree(gpuFrame_);
        gpuFrame_ = nullptr;
    }

    if (gpuInput_) {
        cudaFree(gpuInput_);
        gpuInput_ = nullptr;
    }

    gpuFrameCapacity_ = 0;
}

PredictionType ObjectDetectionEstimator::type() const
{
    return PredictionType::OBJECT_DETECTION;
}

std::string ObjectDetectionEstimator::name() const
{
    return "Object Detection";
}

bool ObjectDetectionEstimator::hasResult() const
{
    return !visualization_.empty();
}

void ObjectDetectionEstimator::predict(
    const cv::Mat& frame
) {
    if (frame.empty()) {

        visualization_.release();
        detectionColorMap_.release();

        return;
    }

    if (frame.channels() != 3 || frame.depth() != CV_8U) {
        throw std::runtime_error(
            "Object detection requires an 8-bit BGR frame."
        );
    }

    checkCuda(
        cudaSetDevice(0),
        "Selecting CUDA device"
    );

    const int frameWidth = frame.cols;
    const int frameHeight = frame.rows;

    const double scale = std::min(
        static_cast<double>(INPUT_WIDTH) / frameWidth,
        static_cast<double>(INPUT_HEIGHT) / frameHeight
    );

    const int resizedWidth =
        static_cast<int>(
            std::round(frameWidth * scale)
        );

    const int resizedHeight =
        static_cast<int>(
            std::round(frameHeight * scale)
        );

    const int padLeft =
        (INPUT_WIDTH - resizedWidth) / 2;

    const int padTop =
        (INPUT_HEIGHT - resizedHeight) / 2;

    ensureGpuBuffers(
        frameWidth,
        frameHeight
    );

    checkCuda(
        cudaMemcpy2D(
            gpuFrame_,
            static_cast<std::size_t>(frameWidth) * 3,
            frame.data,
            frame.step,
            static_cast<std::size_t>(frameWidth) * 3,
            frameHeight,
            cudaMemcpyHostToDevice
        ),
        "Uploading source frame"
    );

    launchDetectionPreprocess(
        gpuFrame_,
        frameWidth,
        frameHeight,
        static_cast<std::size_t>(frameWidth) * 3,
        gpuInput_,
        resizedWidth,
        resizedHeight,
        padLeft,
        padTop,
        nullptr
    );

    checkCuda(
        cudaGetLastError(),
        "Launching preprocessing kernel"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "Synchronizing preprocessing"
    );

    const int64_t inputShape[] = {
        1,
        3,
        INPUT_HEIGHT,
        INPUT_WIDTH
    };

    Ort::MemoryInfo cudaMemoryInfo(
        "Cuda",
        OrtAllocatorType::OrtDeviceAllocator,
        0,
        OrtMemTypeDefault
    );

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            cudaMemoryInfo,
            gpuInput_,
            INPUT_ELEMENTS,
            inputShape,
            4
        );

    Ort::IoBinding binding(session);

    binding.BindInput(
        inputName_.c_str(),
        inputTensor
    );

    binding.BindOutput(
        outputName_.c_str(),
        cudaMemoryInfo
    );

    binding.SynchronizeInputs();

    session.Run(
        Ort::RunOptions{nullptr},
        binding
    );

    binding.SynchronizeOutputs();

    auto outputs = binding.GetOutputValues();

    if (outputs.empty()) {
        throw std::runtime_error(
            "Object detection returned no outputs."
        );
    }

    auto outputInfo =
        outputs[0].GetTensorTypeAndShapeInfo();

    const auto outputShape =
        outputInfo.GetShape();

    if (
        outputShape != std::vector<int64_t>{
            1,
            NUM_DETECTIONS,
            NUM_VALUES
        } ||
        outputInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Unexpected detection output. "
            "Expected [1,300,6] float tensor."
        );
    }

    std::array<float, OUTPUT_ELEMENTS> outputData{};

    checkCuda(
        cudaMemcpy(
            outputData.data(),
            outputs[0].GetTensorData<float>(),
            OUTPUT_ELEMENTS * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "Downloading detection results"
    );

    std::vector<Detection> detections;

    detections.reserve(NUM_DETECTIONS);

    for (int i = 0; i < NUM_DETECTIONS; ++i) {

        const float* values =
            outputData.data() + i * NUM_VALUES;

        const float x1 = values[0];
        const float y1 = values[1];
        const float x2 = values[2];
        const float y2 = values[3];

        const float confidence = values[4];
        const float classValue = values[5];

        if (
            !std::isfinite(x1) ||
            !std::isfinite(y1) ||
            !std::isfinite(x2) ||
            !std::isfinite(y2) ||
            !std::isfinite(confidence) ||
            !std::isfinite(classValue)
        ) {
            continue;
        }

        if (
            confidence < CONFIDENCE_THRESHOLD ||
            classValue < 0.0f ||
            classValue >= NUM_CLASSES
        ) {
            continue;
        }

        const int classId =
            static_cast<int>(
                std::round(classValue)
            );

        if (
            classId < 0 ||
            classId >= NUM_CLASSES
        ) {
            continue;
        }

        const float originalX1 =
            (x1 - padLeft) / scale;

        const float originalY1 =
            (y1 - padTop) / scale;

        const float originalX2 =
            (x2 - padLeft) / scale;

        const float originalY2 =
            (y2 - padTop) / scale;

        const int left = std::clamp(
            static_cast<int>(std::floor(originalX1)),
            0,
            frameWidth
        );

        const int top = std::clamp(
            static_cast<int>(std::floor(originalY1)),
            0,
            frameHeight
        );

        const int right = std::clamp(
            static_cast<int>(std::ceil(originalX2)),
            0,
            frameWidth
        );

        const int bottom = std::clamp(
            static_cast<int>(std::ceil(originalY2)),
            0,
            frameHeight
        );

        cv::Rect box(
            left,
            top,
            right - left,
            bottom - top
        );

        if (box.empty()) {
            continue;
        }

        detections.push_back({
            box,
            classId,
            confidence
        });
    }

    // Sort by confidence so that higher-confidence detections
    // claim overlapping pixels first.

    std::stable_sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        }
    );

    visualization_ = frame.clone();

    // Black represents pixels outside all detected boxes.
    detectionColorMap_ = cv::Mat(
        frame.size(),
        CV_8UC3,
        cv::Scalar(0, 0, 0)
    );

    const auto& colors = getClassColors();

    // Construct pure detection colour map.
    // First detection to claim a pixel wins.

    cv::Mat assigned(
        frame.size(),
        CV_8UC1,
        cv::Scalar(0)
    );

    for (const auto& detection : detections) {

        const cv::Scalar& color =
            colors[detection.classId];

        cv::Mat assignedRegion =
            assigned(detection.box);

        cv::Mat colorRegion =
            detectionColorMap_(detection.box);

        const cv::Mat available =
            assignedRegion == 0;

        colorRegion.setTo(
            color,
            available
        );

        assignedRegion.setTo(
            255,
            available
        );

        // Draw bounding box.

        cv::rectangle(
            visualization_,
            detection.box,
            color,
            2,
            cv::LINE_8
        );

        const std::string label =
            CLASS_NAMES[detection.classId] +
            " " +
            cv::format("%.2f", detection.confidence);

        constexpr double FONT_SCALE = 0.5;
        constexpr int FONT_THICKNESS = 1;

        int baseline = 0;

        const cv::Size textSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                FONT_THICKNESS,
                &baseline
            );

        const int labelTop = std::max(
            0,
            detection.box.y - textSize.height - 8
        );

        cv::rectangle(
            visualization_,
            cv::Point(
                detection.box.x,
                labelTop
            ),
            cv::Point(
                detection.box.x + textSize.width + 8,
                labelTop + textSize.height + 8
            ),
            color,
            cv::FILLED
        );

        cv::putText(
            visualization_,
            label,
            cv::Point(
                detection.box.x + 4,
                labelTop + textSize.height + 2
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            cv::Scalar(0, 0, 0),
            FONT_THICKNESS,
            cv::LINE_8
        );
    }
}

cv::Mat ObjectDetectionEstimator::visualize() const
{
    if (visualization_.empty()) {
        return {};
    }

    return visualization_.clone();
}

cv::Mat ObjectDetectionEstimator::getDetectionColorMap() const
{
    if (detectionColorMap_.empty()) {
        return {};
    }

    return detectionColorMap_.clone();
}
