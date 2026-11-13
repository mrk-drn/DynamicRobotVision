#include "estimators/ObjectDetectionEstimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;

constexpr int NUM_CLASSES = 80;
constexpr int NUM_DETECTIONS = 300;
constexpr int NUM_VALUES = 6;

constexpr float CONFIDENCE_THRESHOLD = 0.05f;
constexpr float NMS_THRESHOLD = 0.45f;

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

const std::array<cv::Scalar, NUM_CLASSES>& getClassColors() {

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

struct Detection {
    cv::Rect box;
    int classId = -1;
    float confidence = 0.0f;
};

}

ObjectDetectionEstimator::ObjectDetectionEstimator(
    const std::string& modelPath
)
    : env(ORT_LOGGING_LEVEL_WARNING, "AdaptivePerception"),
      session(nullptr)
{
    std::cout << "Loading object detection model: "
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

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName =
        session.GetInputNameAllocated(0, allocator);

    auto outputName =
        session.GetOutputNameAllocated(0, allocator);

    inputName_ = inputName.get();
    outputName_ = outputName.get();

    std::cout << "Object detection model loaded."
              << std::endl;

    std::cout << "ONNX Runtime: CUDA GPU 0"
              << std::endl;
}

PredictionType ObjectDetectionEstimator::type() const {
    return PredictionType::OBJECT_DETECTION;
}

std::string ObjectDetectionEstimator::name() const {
    return "Object Detection";
}

bool ObjectDetectionEstimator::hasResult() const {
    return !visualization_.empty();
}

void ObjectDetectionEstimator::predict(
    const cv::Mat& frame
) {
    if (frame.empty()) {
        visualization_.release();
        return;
    }

    // Letterbox preprocessing.

    const double scale = std::min(
        static_cast<double>(INPUT_WIDTH) / frame.cols,
        static_cast<double>(INPUT_HEIGHT) / frame.rows
    );

    const int resizedWidth =
        static_cast<int>(std::round(frame.cols * scale));

    const int resizedHeight =
        static_cast<int>(std::round(frame.rows * scale));

    const int padLeft =
        (INPUT_WIDTH - resizedWidth) / 2;

    const int padTop =
        (INPUT_HEIGHT - resizedHeight) / 2;

    const int padRight =
        INPUT_WIDTH - resizedWidth - padLeft;

    const int padBottom =
        INPUT_HEIGHT - resizedHeight - padTop;

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

    // Prepare input tensor.

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

    // Run inference.

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

    // Validate output: [1, 300, 6].

    auto outputInfo =
        outputs[0].GetTensorTypeAndShapeInfo();

    const auto outputShape = outputInfo.GetShape();

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
            "Expected [1, 300, 6] float tensor."
        );
    }

    const float* outputData =
        outputs[0].GetTensorData<float>();

    // Decode detections.
    //
    // Format:
    // [x1, y1, x2, y2, confidence, class_id]
    //
    // Coordinates are in model input pixels.

    std::vector<Detection> candidates;

    candidates.reserve(NUM_DETECTIONS);

    for (int i = 0; i < NUM_DETECTIONS; ++i) {

        const float* detection =
            outputData + i * NUM_VALUES;

        const float x1 = detection[0];
        const float y1 = detection[1];
        const float x2 = detection[2];
        const float y2 = detection[3];

        const float confidence = detection[4];

        const float classValue = detection[5];

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
            static_cast<int>(std::round(classValue));

        if (classId < 0 || classId >= NUM_CLASSES) {
            continue;
        }

        // Restore original frame coordinates.

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
            frame.cols
        );

        const int top = std::clamp(
            static_cast<int>(std::floor(originalY1)),
            0,
            frame.rows
        );

        const int right = std::clamp(
            static_cast<int>(std::ceil(originalX2)),
            0,
            frame.cols
        );

        const int bottom = std::clamp(
            static_cast<int>(std::ceil(originalY2)),
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

        candidates.push_back({
            box,
            classId,
            confidence
        });
    }

    // Class-aware NMS.

    std::vector<Detection> filtered;

    filtered.reserve(candidates.size());

    for (int classId = 0; classId < NUM_CLASSES; ++classId) {

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> indices;

        for (
            int i = 0;
            i < static_cast<int>(candidates.size());
            ++i
        ) {
            if (candidates[i].classId == classId) {

                boxes.push_back(candidates[i].box);
                scores.push_back(candidates[i].confidence);
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
            filtered.push_back(
                candidates[indices[index]]
            );
        }
    }

    // Sort by confidence.

    std::sort(
        filtered.begin(),
        filtered.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        }
    );

    // Draw detections.

    visualization_ = frame.clone();

    const auto& colors = getClassColors();

    for (const auto& detection : filtered) {

        const cv::Scalar color =
            colors[detection.classId];

        cv::rectangle(
            visualization_,
            detection.box,
            color,
            2,
            cv::LINE_AA
        );

        const std::string label =
            CLASS_NAMES[detection.classId] +
            " " +
            cv::format("%.2f", detection.confidence);

        int baseline = 0;

        const cv::Size textSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
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
            0.5,
            cv::Scalar(0, 0, 0),
            1,
            cv::LINE_AA
        );
    }
}

cv::Mat ObjectDetectionEstimator::visualize() const {

    if (visualization_.empty()) {
        return {};
    }

    return visualization_.clone();
}