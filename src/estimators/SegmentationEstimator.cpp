
#include "estimators/SegmentationEstimator.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace {

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;

constexpr int NUM_CLASSES = 80;
constexpr int NUM_MASK_COEFFS = 32;

constexpr int NUM_FEATURES =
    4 + NUM_CLASSES + NUM_MASK_COEFFS;

constexpr int NUM_PREDICTIONS = 8400;

constexpr int MASK_WIDTH = 160;
constexpr int MASK_HEIGHT = 160;

constexpr int MASK_SIZE =
    MASK_WIDTH * MASK_HEIGHT;

constexpr float CONFIDENCE_THRESHOLD = 0.05f;
constexpr float MASK_THRESHOLD = 0.6f;
constexpr float MASK_OPACITY = 0.5f;

constexpr float NMS_THRESHOLD = 0.90f;

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
                    170,
                    175
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

struct Instance {
    cv::Rect box;
    int classId = -1;
    float confidence = 0.0f;

    std::array<float, NUM_MASK_COEFFS> coefficients{};
};

}

SegmentationEstimator::SegmentationEstimator(
    const std::string& modelPath
)
    : env(ORT_LOGGING_LEVEL_WARNING, "AdaptivePerception"),
      session(nullptr)
{
    std::cout << "Loading segmentation model: "
              << modelPath << std::endl;

    const auto providers = Ort::GetAvailableProviders();

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

    std::cout << "Segmentation model loaded." << std::endl;
    std::cout << "ONNX Runtime: CUDA GPU 0" << std::endl;

    const int cudaDevices =
        cv::cuda::getCudaEnabledDeviceCount();

    if (cudaDevices <= 0) {
        throw std::runtime_error(
            "OpenCV CUDA is unavailable. "
            "Build OpenCV with CUDA support."
        );
    }

    cv::cuda::setDevice(0);

    std::cout << "OpenCV CUDA: GPU 0" << std::endl;
    std::cout << "CUDA segmentation post-processing enabled."
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

void SegmentationEstimator::predict(const cv::Mat& frame)
{
    if (frame.empty()) {
        visualization_.release();
        segmentationMask_.release();
        return;
    }

    const double scale = std::min(
        static_cast<double>(INPUT_WIDTH) / frame.cols,
        static_cast<double>(INPUT_HEIGHT) / frame.rows
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

    const int padLeft =
        (INPUT_WIDTH - resizedWidth) / 2;

    const int padRight =
        INPUT_WIDTH - resizedWidth - padLeft;

    const int padTop =
        (INPUT_HEIGHT - resizedHeight) / 2;

    const int padBottom =
        INPUT_HEIGHT - resizedHeight - padTop;

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

    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed,
        1.0 / 255.0,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

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

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        2
    );

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

    std::vector<Instance> instances;

    instances.reserve(512);

    for (int i = 0; i < NUM_PREDICTIONS; ++i) {
        const float cx = detections[i];

        const float cy =
            detections[NUM_PREDICTIONS + i];

        const float w =
            detections[2 * NUM_PREDICTIONS + i];

        const float h =
            detections[3 * NUM_PREDICTIONS + i];

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

        cv::Rect box(
            left,
            top,
            right - left,
            bottom - top
        );

        if (box.empty()) {
            continue;
        }

        for (int c = 0; c < NUM_CLASSES; ++c) {
            const float score =
                detections[
                    (4 + c) * NUM_PREDICTIONS + i
                ];

            if (
                !std::isfinite(score) ||
                score < CONFIDENCE_THRESHOLD
            ) {
                continue;
            }

            Instance instance;

            instance.box = box;
            instance.classId = c;
            instance.confidence = score;

            for (int m = 0; m < NUM_MASK_COEFFS; ++m) {
                instance.coefficients[m] =
                    detections[
                        (4 + NUM_CLASSES + m) *
                        NUM_PREDICTIONS + i
                    ];
            }

            instances.push_back(std::move(instance));
        }
    }

    std::vector<Instance> filteredInstances;

    filteredInstances.reserve(instances.size());

    for (int classId = 0; classId < NUM_CLASSES; ++classId) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> indices;

        for (
            int i = 0;
            i < static_cast<int>(instances.size());
            ++i
        ) {
            if (instances[i].classId == classId) {
                boxes.push_back(instances[i].box);
                scores.push_back(instances[i].confidence);
                indices.push_back(i);
            }
        }

        if (boxes.empty()) {
            continue;
        }

        std::vector<int> kept;

        cv::dnn::NMSBoxes(
            boxes,
            scores,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            kept
        );

        for (int index : kept) {
            filteredInstances.push_back(
                std::move(instances[indices[index]])
            );
        }
    }

    std::sort(
        filteredInstances.begin(),
        filteredInstances.end(),
        [](const Instance& a, const Instance& b) {
            return a.confidence > b.confidence;
        }
    );

    segmentationMask_ = cv::Mat(
        frame.size(),
        CV_8UC1,
        cv::Scalar(0)
    );

    if (filteredInstances.empty()) {
        visualization_ = frame.clone();
        return;
    }

    cv::Mat prototypeMatrix(
        NUM_MASK_COEFFS,
        MASK_SIZE,
        CV_32F,
        const_cast<float*>(prototypes)
    );

    cv::cuda::GpuMat gpuPrototypes;

    gpuPrototypes.upload(prototypeMatrix);

    cv::cuda::GpuMat gpuFrame;

    gpuFrame.upload(frame);

    cv::cuda::GpuMat gpuColorLayer;

    gpuFrame.copyTo(gpuColorLayer);

    cv::cuda::GpuMat gpuCombinedMask(
        frame.size(),
        CV_8UC1,
        cv::Scalar(0)
    );

    cv::cuda::GpuMat gpuVisualization;

    gpuFrame.copyTo(gpuVisualization);

    cv::cuda::GpuMat gpuMaskLogits;
    cv::cuda::GpuMat gpuProbabilities;
    cv::cuda::GpuMat gpuFullMask;
    cv::cuda::GpuMat gpuCroppedMask;
    cv::cuda::GpuMat gpuOriginalMask;
    cv::cuda::GpuMat gpuBinaryMask;

    cv::cuda::GpuMat gpuColorROI;
    cv::cuda::GpuMat gpuCombinedROI;

    const auto& classColors = getClassColors();

    for (const auto& instance : filteredInstances) {
        cv::Mat coefficients(
            1,
            NUM_MASK_COEFFS,
            CV_32F,
            const_cast<float*>(
                instance.coefficients.data()
            )
        );

        cv::cuda::GpuMat gpuCoefficients;

        gpuCoefficients.upload(coefficients);

        cv::cuda::GpuMat gpuMaskLogitsFlat;

        cv::cuda::gemm(
            gpuCoefficients,
            gpuPrototypes,
            1.0,
            cv::cuda::GpuMat(),
            0.0,
            gpuMaskLogitsFlat
        );

        gpuMaskLogits = gpuMaskLogitsFlat.reshape(
            1,
            MASK_HEIGHT
        );

        cv::cuda::GpuMat gpuNegativeMaskLogits;

        cv::cuda::multiply(
            gpuMaskLogits,
            cv::Scalar::all(-1.0),
            gpuNegativeMaskLogits
        );

        cv::cuda::exp(
            gpuNegativeMaskLogits,
            gpuProbabilities
        );

        cv::cuda::add(
            gpuProbabilities,
            cv::Scalar(1.0),
            gpuProbabilities
        );

        cv::cuda::divide(
            cv::Scalar(1.0),
            gpuProbabilities,
            gpuProbabilities
        );

        cv::cuda::resize(
            gpuProbabilities,
            gpuFullMask,
            cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        gpuCroppedMask = gpuFullMask(
            cv::Rect(
                padLeft,
                padTop,
                resizedWidth,
                resizedHeight
            )
        );

        cv::cuda::resize(
            gpuCroppedMask,
            gpuOriginalMask,
            frame.size(),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        cv::cuda::GpuMat gpuInstanceROI =
            gpuOriginalMask(instance.box);

        cv::cuda::compare(
            gpuInstanceROI,
            cv::Scalar(MASK_THRESHOLD),
            gpuBinaryMask,
            cv::CMP_GT
        );

        gpuColorROI = gpuColorLayer(instance.box);

        gpuColorROI.setTo(
            classColors[instance.classId],
            gpuBinaryMask
        );

        gpuCombinedROI = gpuCombinedMask(instance.box);

        gpuCombinedROI.setTo(
            cv::Scalar(255),
            gpuBinaryMask
        );
    }

    cv::cuda::GpuMat gpuBlended;

    cv::cuda::addWeighted(
        gpuFrame,
        1.0 - MASK_OPACITY,
        gpuColorLayer,
        MASK_OPACITY,
        0.0,
        gpuBlended
    );

    gpuBlended.copyTo(
        gpuVisualization,
        gpuCombinedMask
    );

    gpuVisualization.download(visualization_);

    gpuCombinedMask.download(segmentationMask_);
}

cv::Mat SegmentationEstimator::visualize() const
{
    if (visualization_.empty()) {
        return {};
    }

    return visualization_.clone();
}

cv::Mat SegmentationEstimator::getSegmentationMask() const
{
    if (segmentationMask_.empty()) {
        return {};
    }

    return segmentationMask_.clone();
}
