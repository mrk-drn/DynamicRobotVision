#include "DepthEstimator.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>

DepthEstimator::DepthEstimator(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "AdaptivePerception"),
      session(nullptr)
{
    std::cout << "Loading depth model: "
              << modelPath << std::endl;

    // Check CUDA availability
    const auto providers = Ort::GetAvailableProviders();

    const bool cudaAvailable =
        std::find(
            providers.begin(),
            providers.end(),
            "CUDAExecutionProvider"
        ) != providers.end();

    if (!cudaAvailable) {
        throw std::runtime_error(
            "CUDAExecutionProvider is unavailable. "
            "Check your ONNX Runtime GPU installation."
        );
    }

    // Configure ONNX Runtime
    Ort::SessionOptions sessionOptions;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    // Enable NVIDIA GPU
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;

    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

    // Create inference session
    session = Ort::Session(
        env,
        modelPath.c_str(),
        sessionOptions
    );

    std::cout << "Depth model loaded successfully!" << std::endl;
    std::cout << "Execution Provider: CUDA (GPU 0)" << std::endl;
}

cv::Mat DepthEstimator::estimateDepth(const cv::Mat& frame) {

    if (frame.empty()) {
        return cv::Mat();
    }

    // -------------------------------------------------
    // 1. Letterbox preprocessing
    // -------------------------------------------------

    const double scale = std::min(
        static_cast<double>(inputWidth) / frame.cols,
        static_cast<double>(inputHeight) / frame.rows
    );

    const int resizedWidth =
        static_cast<int>(std::round(frame.cols * scale));

    const int resizedHeight =
        static_cast<int>(std::round(frame.rows * scale));

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(resizedWidth, resizedHeight),
        0,
        0,
        cv::INTER_LINEAR
    );

    const int padLeft = (inputWidth - resizedWidth) / 2;
    const int padRight = inputWidth - resizedWidth - padLeft;

    const int padTop = (inputHeight - resizedHeight) / 2;
    const int padBottom = inputHeight - resizedHeight - padTop;

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

    // BGR -> RGB, normalize to [0,1], HWC -> NCHW
    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    // -------------------------------------------------
    // 2. Prepare ONNX input tensor
    // -------------------------------------------------

    std::vector<int64_t> inputShape = {
        1, 3, inputHeight, inputWidth
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

    // -------------------------------------------------
    // 3. Run GPU inference
    // -------------------------------------------------

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session.GetInputNameAllocated(0, allocator);

    auto outputName =
        session.GetOutputNameAllocated(0, allocator);

    const char* inputNames[] = {
        inputName.get()
    };

    const char* outputNames[] = {
        outputName.get()
    };

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    // -------------------------------------------------
    // 4. Extract depth map
    // -------------------------------------------------

    auto& outputTensor = outputTensors.front();

    auto outputInfo =
        outputTensor.GetTensorTypeAndShapeInfo();

    const auto outputShape = outputInfo.GetShape();

    if (outputShape.size() != 4 ||
        outputShape[0] != 1 ||
        outputShape[1] != 1 ||
        outputShape[2] != inputHeight ||
        outputShape[3] != inputWidth ||
        outputInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {

        throw std::runtime_error(
            "Unexpected depth model output shape or type."
        );
    }

    float* outputData =
        outputTensor.GetTensorMutableData<float>();

    cv::Mat depthModel(
        inputHeight,
        inputWidth,
        CV_32F,
        outputData
    );

    // Clone because the ONNX output tensor owns its memory.
    cv::Mat depth = depthModel.clone();

    // -------------------------------------------------
    // 5. Remove letterbox padding
    // -------------------------------------------------

    cv::Rect validRegion(
        padLeft,
        padTop,
        resizedWidth,
        resizedHeight
    );

    cv::Mat croppedDepth = depth(validRegion);

    // Restore original image dimensions
    cv::Mat finalDepth;

    cv::resize(
        croppedDepth,
        finalDepth,
        frame.size(),
        0,
        0,
        cv::INTER_LINEAR
    );

    return finalDepth;
}
