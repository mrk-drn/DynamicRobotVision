#include "estimators/DepthEstimator.hpp"
#include "estimators/DepthKernels.cuh"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double NORMALIZATION_SCALE = 1.0 / 255.0;
constexpr float DEPTH_PERCENTILE = 0.05f;
constexpr int LETTERBOX_VALUE = 114;

}

DepthEstimator::DepthEstimator(
    const std::string& modelPath
)
    : env(
        ORT_LOGGING_LEVEL_WARNING,
        "AdaptivePerception"
    ),
      session(nullptr)
{
    std::cout
        << "Loading depth model: "
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

    Ort::SessionOptions options;

    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    options.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);

    OrtCUDAProviderOptions cudaOptions{};

    cudaOptions.device_id = 0;

    cudaOptions.cudnn_conv_algo_search =
        OrtCudnnConvAlgoSearchHeuristic;

    cudaOptions.do_copy_in_default_stream = 1;

    options.AppendExecutionProvider_CUDA(
        cudaOptions
    );

    session = Ort::Session(
        env,
        modelPath.c_str(),
        options
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

    letterboxed_.create(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        CV_8UC3
    );

    blob_.create(
        1,
        static_cast<int>(INPUT_ELEMENTS),
        CV_32F
    );

    std::cout
        << "Depth model loaded."
        << std::endl;

    std::cout
        << "ONNX Runtime: CUDA GPU 0"
        << std::endl;

    std::cout
        << "Depth CUDA post-processing enabled."
        << std::endl;
}

void DepthEstimator::predict(
    const cv::Mat& frame
) {
    if (frame.empty()) {
        depthResult_.release();
        depthVisualization_.release();

        nearestDistanceCache_ = -1.0f;
        nearestDistanceValid_ = false;

        return;
    }

    if (frame.type() != CV_8UC3) {
        throw std::runtime_error(
            "DepthEstimator expects CV_8UC3 BGR frames."
        );
    }

    const double scale = std::min(
        static_cast<double>(INPUT_WIDTH) / frame.cols,
        static_cast<double>(INPUT_HEIGHT) / frame.rows
    );

    const int resizedWidth = std::clamp(
        static_cast<int>(
            std::round(frame.cols * scale)
        ),
        1,
        INPUT_WIDTH
    );

    const int resizedHeight = std::clamp(
        static_cast<int>(
            std::round(frame.rows * scale)
        ),
        1,
        INPUT_HEIGHT
    );

    const int padLeft =
        (INPUT_WIDTH - resizedWidth) / 2;

    const int padTop =
        (INPUT_HEIGHT - resizedHeight) / 2;

    const int padRight =
        INPUT_WIDTH - resizedWidth - padLeft;

    const int padBottom =
        INPUT_HEIGHT - resizedHeight - padTop;

    cv::resize(
        frame,
        resized_,
        cv::Size(
            resizedWidth,
            resizedHeight
        ),
        0,
        0,
        cv::INTER_LINEAR
    );

    letterboxed_.setTo(
        cv::Scalar(
            LETTERBOX_VALUE,
            LETTERBOX_VALUE,
            LETTERBOX_VALUE
        )
    );

    resized_.copyTo(
        letterboxed_(
            cv::Rect(
                padLeft,
                padTop,
                resizedWidth,
                resizedHeight
            )
        )
    );

    cv::dnn::blobFromImage(
        letterboxed_,
        blob_,
        NORMALIZATION_SCALE,
        cv::Size(
            INPUT_WIDTH,
            INPUT_HEIGHT
        ),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo_,
            blob_.ptr<float>(),
            INPUT_ELEMENTS,
            inputShape_,
            4
        );

    const char* inputNames[] = {
        inputName_.c_str()
    };

    const char* outputNames[] = {
        outputName_.c_str()
    };

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    if (outputs.empty()) {
        throw std::runtime_error(
            "Depth model returned no output."
        );
    }

    auto outputInfo =
        outputs[0].GetTensorTypeAndShapeInfo();

    const auto outputShape =
        outputInfo.GetShape();

    if (
        outputShape.size() != 4 ||
        outputShape[0] != 1 ||
        outputShape[1] != 1 ||
        outputShape[2] != INPUT_HEIGHT ||
        outputShape[3] != INPUT_WIDTH ||
        outputInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Unexpected depth model output shape or type."
        );
    }

    const float* outputData =
        outputs[0].GetTensorData<float>();

    cv::Mat depthModel(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        CV_32F,
        const_cast<float*>(outputData)
    );

    const cv::Rect validRegion(
        padLeft,
        padTop,
        resizedWidth,
        resizedHeight
    );

    cv::Mat croppedDepth =
        depthModel(validRegion);

    cv::resize(
        croppedDepth,
        depthResult_,
        frame.size(),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    nearestDistanceValid_ = false;
    nearestDistanceCache_ = -1.0f;

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

cv::Mat DepthEstimator::visualize() const
{
    if (depthVisualization_.empty()) {
        return {};
    }

    return depthVisualization_.clone();
}

bool DepthEstimator::hasResult() const
{
    return !depthResult_.empty();
}

const cv::Mat& DepthEstimator::getDepth() const
{
    return depthResult_;
}

float DepthEstimator::nearestDistanceMeters() const
{
    if (
        depthResult_.empty() ||
        depthResult_.type() != CV_32F
    ) {
        return -1.0f;
    }

    if (nearestDistanceValid_) {
        return nearestDistanceCache_;
    }

    nearestDistanceCache_ =
        DepthKernels::calculateNearestPercentile(
            depthResult_.ptr<float>(),
            depthResult_.total(),
            DEPTH_PERCENTILE
        );

    nearestDistanceValid_ = true;

    return nearestDistanceCache_;
}
