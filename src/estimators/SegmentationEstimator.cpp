
#include "estimators/SegmentationEstimator.hpp"

#include <cuda_runtime.h>

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <utility>

namespace {

constexpr int INPUT_WIDTH = 768;
constexpr int INPUT_HEIGHT = 768;

constexpr int NUM_CLASSES = 80;
constexpr int NUM_MASK_COEFFS = 32;

constexpr int NUM_FEATURES =
    4 + NUM_CLASSES + NUM_MASK_COEFFS;

constexpr int NUM_PREDICTIONS = 12096;

constexpr int MASK_WIDTH = 192;
constexpr int MASK_HEIGHT = 192;

constexpr float CONFIDENCE_SCORE = 0.05f;
constexpr float MASK_THRESHOLD = 0.5f;
constexpr float MASK_OPACITY = 0.5f;

constexpr float NMS_THRESHOLD = 0.90f;

constexpr int MAX_CANDIDATES = NUM_PREDICTIONS;

const std::array<cv::Scalar, NUM_CLASSES>&
getClassColors()
{
    static const std::array<cv::Scalar, NUM_CLASSES> colors =
        [] {

            std::array<cv::Scalar, NUM_CLASSES> result;

            cv::Mat hsv(
                1,
                1,
                CV_8UC3
            );

            cv::Mat bgr;

            for (int i = 0; i < NUM_CLASSES; ++i) {

                hsv.at<cv::Vec3b>(0, 0) =
                    cv::Vec3b(
                        static_cast<unsigned char>(
                            (i * 137) % 180
                        ),
                        170,
                        175
                    );

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

struct Instance {

    cv::Rect box;

    int classId = -1;

    float confidence = 0.0f;

    std::array<float, NUM_MASK_COEFFS>
        coefficients{};
};

} // namespace


SegmentationEstimator::SegmentationEstimator(
    const std::string& modelPath
)
    : env(
        ORT_LOGGING_LEVEL_WARNING,
        "AdaptivePerception"
    ),
    session(nullptr)
{
    std::cout
        << "Loading segmentation model: "
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

    // ONNX Runtime CUDA inference.

    sessionOptions.AppendExecutionProvider_CUDA(
        cudaOptions
    );

    session = Ort::Session(
        env,
        modelPath.c_str(),
        sessionOptions
    );

    std::cout
        << "Segmentation model loaded."
        << std::endl;

    std::cout
        << "ONNX Runtime: CUDA GPU 0"
        << std::endl;

    // Initialize CUDA device for custom kernels.

    const cudaError_t cudaStatus =
        cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "Could not initialize CUDA GPU 0: "
            ) +
            cudaGetErrorString(cudaStatus)
        );
    }

    std::cout
        << "Custom CUDA segmentation kernels initialized."
        << std::endl;

    std::cout
        << "Batched mask generation enabled."
        << std::endl;

    std::cout
        << "GPU mask composition and blending enabled."
        << std::endl;
}


PredictionType SegmentationEstimator::type() const
{
    return PredictionType::SEGMENTATION;
}


std::string SegmentationEstimator::name() const
{
    return "Segmentation";
}


bool SegmentationEstimator::hasResult() const
{
    return !visualization_.empty();
}


void SegmentationEstimator::predict(
    const cv::Mat& frame
) {
    // Handle empty input.

    if (frame.empty()) {

        visualization_.release();
        segmentationMask_.release();
        segmentationColorMap_.release();

        return;
    }

    if (frame.type() != CV_8UC3) {
        throw std::runtime_error(
            "SegmentationEstimator expects CV_8UC3 input."
        );
    }

    // Calculate letterbox dimensions.

    const float scale = static_cast<float>(
        std::min(
            static_cast<double>(INPUT_WIDTH) / frame.cols,
            static_cast<double>(INPUT_HEIGHT) / frame.rows
        )
    );

    const int resizedWidth =
        static_cast<int>(
            std::round(frame.cols * scale)
        );

    const int resizedHeight =
        static_cast<int>(
            std::round(frame.rows * scale)
        );

    const int padLeft =
        (INPUT_WIDTH - resizedWidth) / 2;

    const int padTop =
        (INPUT_HEIGHT - resizedHeight) / 2;

    const int padRight =
        INPUT_WIDTH - resizedWidth - padLeft;

    const int padBottom =
        INPUT_HEIGHT - resizedHeight - padTop;

    // Resize input.

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            resizedWidth,
            resizedHeight
        ),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    // Letterbox input.

    cv::Mat letterboxed;

    cv::copyMakeBorder(
        resized,
        letterboxed,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            114,
            114,
            114
        )
    );

    // Convert input into NCHW float blob.

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed,
        1.0 / 255.0,
        cv::Size(
            INPUT_WIDTH,
            INPUT_HEIGHT
        ),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    // Create ONNX input tensor.

    const int64_t inputShape[] = {
        1,
        3,
        INPUT_HEIGHT,
        INPUT_WIDTH
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
            inputShape,
            4
        );

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session.GetInputNameAllocated(
            0,
            allocator
        );

    const char* inputNames[] = {
        inputName.get()
    };

    const char* outputNames[] = {
        "output0",
        "output1"
    };

    // Run ONNX inference.

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        2
    );

    // Validate detection output.

    auto detectionInfo =
        outputs[0].GetTensorTypeAndShapeInfo();

    const auto detectionShape =
        detectionInfo.GetShape();

    if (
        detectionShape != std::vector<int64_t>{
            1,
            NUM_FEATURES,
            NUM_PREDICTIONS
        } ||
        detectionInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Unexpected segmentation detection output."
        );
    }

    const float* detections =
        outputs[0].GetTensorData<float>();

    // Validate prototype output.

    auto prototypeInfo =
        outputs[1].GetTensorTypeAndShapeInfo();

    const auto prototypeShape =
        prototypeInfo.GetShape();

    if (
        prototypeShape != std::vector<int64_t>{
            1,
            NUM_MASK_COEFFS,
            MASK_HEIGHT,
            MASK_WIDTH
        } ||
        prototypeInfo.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Unexpected segmentation prototype output."
        );
    }

    const float* prototypes =
        outputs[1].GetTensorData<float>();

    // Decode detections.

    std::vector<Instance> instances;

    instances.reserve(MAX_CANDIDATES);

    for (int i = 0; i < NUM_PREDICTIONS; ++i) {

        const float cx =
            detections[i];

        const float cy =
            detections[
                NUM_PREDICTIONS + i
            ];

        const float w =
            detections[
                2 * NUM_PREDICTIONS + i
            ];

        const float h =
            detections[
                3 * NUM_PREDICTIONS + i
            ];

        if (
            !std::isfinite(cx) ||
            !std::isfinite(cy) ||
            !std::isfinite(w) ||
            !std::isfinite(h) ||
            w <= 0.0f ||
            h <= 0.0f
        ) {
            continue;
        }

        // Convert letterbox coordinates to frame coordinates.

        const float x1 =
            (cx - w * 0.5f - padLeft) / scale;

        const float y1 =
            (cy - h * 0.5f - padTop) / scale;

        const float x2 =
            (cx + w * 0.5f - padLeft) / scale;

        const float y2 =
            (cy + h * 0.5f - padTop) / scale;

        const int left = std::clamp(
            static_cast<int>(std::floor(x1)),
            0,
            frame.cols
        );

        const int top = std::clamp(
            static_cast<int>(std::floor(y1)),
            0,
            frame.rows
        );

        const int right = std::clamp(
            static_cast<int>(std::ceil(x2)),
            0,
            frame.cols
        );

        const int bottom = std::clamp(
            static_cast<int>(std::ceil(y2)),
            0,
            frame.rows
        );

        const cv::Rect box(
            left,
            top,
            right - left,
            bottom - top
        );

        if (box.empty()) {
            continue;
        }

        // Decode class confidence scores.

        for (int c = 0; c < NUM_CLASSES; ++c) {

            const float score =
                detections[
                    (4 + c) *
                    NUM_PREDICTIONS + i
                ];

            if (
                !std::isfinite(score) ||
                score < CONFIDENCE_SCORE
            ) {
                continue;
            }

            Instance instance;

            instance.box = box;
            instance.classId = c;
            instance.confidence = score;

            // Copy all 32 mask coefficients.

            for (int m = 0; m < NUM_MASK_COEFFS; ++m) {

                instance.coefficients[m] =
                    detections[
                        (4 + NUM_CLASSES + m) *
                        NUM_PREDICTIONS + i
                    ];
            }

            instances.push_back(
                std::move(instance)
            );
        }
    }

    // Class-wise Non-Maximum Suppression.

    std::vector<Instance> filteredInstances;

    filteredInstances.reserve(instances.size());

    for (int classId = 0;
         classId < NUM_CLASSES;
         ++classId) {

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> indices;

        boxes.reserve(instances.size() / NUM_CLASSES + 1);
        scores.reserve(instances.size() / NUM_CLASSES + 1);
        indices.reserve(instances.size() / NUM_CLASSES + 1);

        for (
            int i = 0;
            i < static_cast<int>(instances.size());
            ++i
        ) {

            const Instance& instance =
                instances[i];

            if (instance.classId != classId) {
                continue;
            }

            boxes.push_back(instance.box);
            scores.push_back(instance.confidence);
            indices.push_back(i);
        }

        if (boxes.empty()) {
            continue;
        }

        std::vector<int> kept;

        cv::dnn::NMSBoxes(
            boxes,
            scores,
            CONFIDENCE_SCORE,
            NMS_THRESHOLD,
            kept
        );

        for (const int index : kept) {

            filteredInstances.push_back(
                std::move(
                    instances[indices[index]]
                )
            );
        }
    }

    // Sort by descending confidence.

    std::sort(
        filteredInstances.begin(),
        filteredInstances.end(),
        [](
            const Instance& a,
            const Instance& b
        ) {
            return a.confidence > b.confidence;
        }
    );

    // Handle frames with no detected instances.

    if (filteredInstances.empty()) {

        visualization_ = frame.clone();

        segmentationMask_ = cv::Mat(
            frame.size(),
            CV_8UC1,
            cv::Scalar(0)
        );

        segmentationColorMap_ = cv::Mat(
            frame.size(),
            CV_8UC3,
            cv::Scalar(0, 0, 0)
        );

        return;
    }

    const int instanceCount =
        static_cast<int>(
            filteredInstances.size()
        );

    // Prepare all instance coefficients in one contiguous matrix.

    cv::Mat coefficientMatrix(
        instanceCount,
        NUM_MASK_COEFFS,
        CV_32FC1
    );

    std::vector<cv::Vec4i> boxes;

    std::vector<cv::Vec3f> colors;

    boxes.reserve(instanceCount);
    colors.reserve(instanceCount);

    const auto& classColors =
        getClassColors();

    for (int i = 0; i < instanceCount; ++i) {

        const Instance& instance =
            filteredInstances[i];

        // Coefficients.

        std::copy(
            instance.coefficients.begin(),
            instance.coefficients.end(),
            coefficientMatrix.ptr<float>(i)
        );

        // Bounding boxes.

        boxes.emplace_back(
            instance.box.x,
            instance.box.y,
            instance.box.width,
            instance.box.height
        );

        // BGR colors.

        const cv::Scalar& color =
            classColors[instance.classId];

        colors.emplace_back(
            static_cast<float>(color[0]),
            static_cast<float>(color[1]),
            static_cast<float>(color[2])
        );
    }

    // Execute the complete batched CUDA pipeline.

    cudaProcessor_.process(
        frame,
        prototypes,
        coefficientMatrix,
        boxes,
        colors,
        resizedWidth,
        resizedHeight,
        padLeft,
        padTop,
        MASK_THRESHOLD,
        MASK_OPACITY,
        visualization_,
        segmentationMask_,
        segmentationColorMap_
    );
}


// Return blended visualization.

cv::Mat SegmentationEstimator::visualize() const
{
    if (visualization_.empty()) {
        return {};
    }

    return visualization_.clone();
}


// Return binary segmentation mask.

cv::Mat SegmentationEstimator::getSegmentationMask() const
{
    if (segmentationMask_.empty()) {
        return {};
    }

    return segmentationMask_.clone();
}


// Return pure segmentation color map.

cv::Mat SegmentationEstimator::getSegmentationColorMap() const
{
    if (segmentationColorMap_.empty()) {
        return {};
    }

    return segmentationColorMap_.clone();
}
