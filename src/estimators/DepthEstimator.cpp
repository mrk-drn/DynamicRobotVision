#include "estimators/DepthEstimator.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>


DepthEstimator::DepthEstimator(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "AdaptivePerception"),
      session(nullptr)
{
    std::cout << "Loading depth model: "
              << modelPath << std::endl;

    // Verify CUDA provider.

    const auto providers = Ort::GetAvailableProviders();

    const bool cudaAvailable =
        std::find(
            providers.begin(),
            providers.end(),
            "CUDAExecutionProvider"
        ) != providers.end();

    if (!cudaAvailable) {
        throw std::runtime_error(
            "CUDAExecutionProvider is unavailable."
        );
    }

    // Configure ONNX Runtime.

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

    // Load model.

    session = Ort::Session(
        env,
        modelPath.c_str(),
        sessionOptions
    );

    // Cache tensor names.

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAllocated =
        session.GetInputNameAllocated(0, allocator);

    auto outputNameAllocated =
        session.GetOutputNameAllocated(0, allocator);

    inputName_ = inputNameAllocated.get();
    outputName_ = outputNameAllocated.get();

    std::cout << "Depth model loaded successfully!"
              << std::endl;

    std::cout << "Execution Provider: CUDA (GPU 0)"
              << std::endl;
}


// --------------------------------------------------
// Prediction
// --------------------------------------------------

void DepthEstimator::predict(const cv::Mat& frame) {

    if (frame.empty()) {

        depthResult_.release();
        depthVisualization_.release();

        return;
    }


    // --------------------------------------------------
    // Letterbox preprocessing
    // --------------------------------------------------

    const double scale = std::min(
        static_cast<double>(inputWidth) / frame.cols,
        static_cast<double>(inputHeight) / frame.rows
    );

    const int resizedWidth =
        static_cast<int>(std::round(frame.cols * scale));

    const int resizedHeight =
        static_cast<int>(std::round(frame.rows * scale));


    // Correct padding calculations.

    const int padLeft =
        (inputWidth - resizedWidth) / 2;

    const int padRight =
        inputWidth - resizedWidth - padLeft;

    const int padTop =
        (inputHeight - resizedHeight) / 2;

    const int padBottom =
        inputHeight - resizedHeight - padTop;


    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(resizedWidth, resizedHeight),
        0,
        0,
        cv::INTER_LINEAR
    );


    cv::Mat letterboxed;

    cv::copyMakeBorder(
        resized,
        letterboxed,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );


    // Validate preprocessing dimensions.

    if (
        letterboxed.cols != inputWidth ||
        letterboxed.rows != inputHeight
    ) {
        throw std::runtime_error(
            "Incorrect letterbox dimensions."
        );
    }


    // --------------------------------------------------
    // Prepare input tensor
    // --------------------------------------------------

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );


    const std::array<int64_t, 4> inputShape = {
        1,
        3,
        inputHeight,
        inputWidth
    };


    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );


    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            blob.ptr<float>(),
            blob.total(),
            inputShape.data(),
            inputShape.size()
        );


    // --------------------------------------------------
    // Run ONNX inference on CUDA
    // --------------------------------------------------

    const char* inputNames[] = {
        inputName_.c_str()
    };

    const char* outputNames[] = {
        outputName_.c_str()
    };


    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );


    // --------------------------------------------------
    // Validate model output
    // --------------------------------------------------

    auto& outputTensor = outputTensors.front();

    auto outputInfo =
        outputTensor.GetTensorTypeAndShapeInfo();

    const auto outputShape = outputInfo.GetShape();

    if (
        outputShape.size() != 4 ||
        outputShape[0] != 1 ||
        outputShape[1] != 1 ||
        outputShape[2] != inputHeight ||
        outputShape[3] != inputWidth ||
        outputInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Unexpected depth model output shape or type."
        );
    }


    // --------------------------------------------------
    // Access output without cloning
    // --------------------------------------------------

    const float* outputData =
        outputTensor.GetTensorData<float>();


    cv::Mat depthModel(
        inputHeight,
        inputWidth,
        CV_32F,
        const_cast<float*>(outputData)
    );


    // --------------------------------------------------
    // Remove letterbox padding
    // --------------------------------------------------

    const cv::Rect validRegion(
        padLeft,
        padTop,
        resizedWidth,
        resizedHeight
    );


    cv::Mat croppedDepth =
        depthModel(validRegion);


    // --------------------------------------------------
    // Restore original frame dimensions
    // --------------------------------------------------

    cv::resize(
        croppedDepth,
        depthResult_,
        frame.size(),
        0,
        0,
        cv::INTER_LINEAR
    );


    // --------------------------------------------------
    // Generate grayscale visualization
    // --------------------------------------------------

    cv::Mat normalized;

    cv::normalize(
        depthResult_,
        normalized,
        0,
        255,
        cv::NORM_MINMAX,
        CV_8UC1
    );


    cv::cvtColor(
        normalized,
        depthVisualization_,
        cv::COLOR_GRAY2BGR
    );
}


// --------------------------------------------------
// Visualization
// --------------------------------------------------

cv::Mat DepthEstimator::visualize() const {

    if (depthVisualization_.empty()) {
        return {};
    }

    return depthVisualization_.clone();
}


// --------------------------------------------------
// Result availability
// --------------------------------------------------

bool DepthEstimator::hasResult() const {

    return !depthResult_.empty();
}


// --------------------------------------------------
// Raw depth access
// --------------------------------------------------

const cv::Mat& DepthEstimator::getDepth() const {

    return depthResult_;
}


// --------------------------------------------------
// Nearest distance estimation
// --------------------------------------------------

float DepthEstimator::nearestDistanceMeters() const {

    if (
        depthResult_.empty() ||
        depthResult_.type() != CV_32F
    ) {
        return -1.0f;
    }


    // Reuse allocated memory between calls.

    static thread_local std::vector<float> values;

    values.clear();

    if (values.capacity() < depthResult_.total()) {

        values.reserve(depthResult_.total());
    }


    // Collect valid depth values.

    for (int y = 0; y < depthResult_.rows; ++y) {

        const float* row =
            depthResult_.ptr<float>(y);

        for (int x = 0; x < depthResult_.cols; ++x) {

            const float value = row[x];

            if (
                std::isfinite(value) &&
                value > 0.0f
            ) {
                values.push_back(value);
            }
        }
    }


    if (values.empty()) {
        return -1.0f;
    }


    // Calculate 5th percentile.

    const size_t index = values.size() / 20;

    std::nth_element(
        values.begin(),
        values.begin() + index,
        values.end()
    );


    return values[index];
}