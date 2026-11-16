#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "core/PredictionManager.hpp"
#include "estimators/DepthEstimator.hpp"
#include "estimators/MotionEstimator.hpp"
#include "estimators/SegmentationEstimator.hpp"
#include "estimators/ObjectDetectionEstimator.hpp"
#include "visualization/Visualizer.hpp"
#include "visualization/PointCloudVisualizer.hpp"

namespace
{

constexpr double PREDICTION_FPS = 15.0;

constexpr int OUTPUT_WIDTH = 1280;
constexpr int OUTPUT_HEIGHT = 720;

constexpr double MOTION_THRESHOLD = 2.0;

constexpr float CAMERA_FX = 600.0f;
constexpr float CAMERA_FY = 600.0f;
constexpr float CAMERA_CX = 640.0f;
constexpr float CAMERA_CY = 360.0f;

constexpr int POINT_CLOUD_SAMPLING_STEP = 10;

constexpr int MASK_THRESHOLD = 12;
constexpr int MIN_MASK_PIXELS = 30;

const std::string WINDOW_NAME =
    "AdaptiveRobotPerception";

using Clock = std::chrono::steady_clock;

const auto PREDICTION_INTERVAL =
    std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(
            1.0 / PREDICTION_FPS
        )
    );


cv::Mat fitToCanvas(
    const cv::Mat& source,
    const cv::Size& canvasSize,
    const cv::Scalar& background = cv::Scalar(25, 25, 25)
)
{
    cv::Mat canvas(
        canvasSize,
        CV_8UC3,
        background
    );

    if (source.empty())
        return canvas;

    cv::Mat converted;

    if (source.channels() == 1)
    {
        cv::cvtColor(
            source,
            converted,
            cv::COLOR_GRAY2BGR
        );
    }
    else if (source.channels() == 4)
    {
        cv::cvtColor(
            source,
            converted,
            cv::COLOR_BGRA2BGR
        );
    }
    else
    {
        converted = source;
    }

    if (converted.type() != CV_8UC3)
    {
        converted.convertTo(converted, CV_8UC3);
    }

    const double scale = std::min(
        static_cast<double>(canvasSize.width) / converted.cols,
        static_cast<double>(canvasSize.height) / converted.rows
    );

    const int width = std::max(
        1,
        static_cast<int>(std::round(converted.cols * scale))
    );

    const int height = std::max(
        1,
        static_cast<int>(std::round(converted.rows * scale))
    );

    cv::Mat resized;

    cv::resize(
        converted,
        resized,
        cv::Size(width, height),
        0,
        0,
        cv::INTER_LINEAR
    );

    const int x = (canvasSize.width - width) / 2;
    const int y = (canvasSize.height - height) / 2;

    resized.copyTo(
        canvas(cv::Rect(x, y, width, height))
    );

    return canvas;
}


cv::Mat createForegroundMask(
    const cv::Mat& original,
    const cv::Mat& prediction,
    const std::string& predictionType
)
{
    if (
        original.empty() ||
        prediction.empty() ||
        original.size() != prediction.size()
    )
    {
        return {};
    }

    cv::Mat difference;
    cv::Mat grayscale;
    cv::Mat mask;

    cv::absdiff(original, prediction, difference);

    cv::cvtColor(
        difference,
        grayscale,
        cv::COLOR_BGR2GRAY
    );

    cv::threshold(
        grayscale,
        mask,
        MASK_THRESHOLD,
        255,
        cv::THRESH_BINARY
    );

    if (predictionType == "Object Detection")
    {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(9, 9)
        );

        cv::morphologyEx(
            mask,
            mask,
            cv::MORPH_CLOSE,
            kernel
        );

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(
            mask,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        cv::Mat filled = cv::Mat::zeros(
            mask.size(),
            CV_8UC1
        );

        for (const auto& contour : contours)
        {
            if (cv::contourArea(contour) < 10.0)
                continue;

            cv::drawContours(
                filled,
                std::vector<std::vector<cv::Point>>{contour},
                0,
                cv::Scalar(255),
                cv::FILLED
            );
        }

        mask = filled;
    }
    else
    {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(3, 3)
        );

        cv::morphologyEx(
            mask,
            mask,
            cv::MORPH_OPEN,
            kernel
        );
    }

    return mask;
}


cv::Mat createOutput(
    const cv::Mat& frame,
    const cv::Mat& prediction,
    const cv::Mat& pointCloud
)
{
    constexpr int PANEL_GAP = 4;

    const int panelWidth =
        (OUTPUT_WIDTH - 2 * PANEL_GAP) / 3;

    const cv::Size panelSize(
        panelWidth,
        OUTPUT_HEIGHT
    );

    cv::Mat originalPanel = fitToCanvas(frame, panelSize);
    cv::Mat predictionPanel = fitToCanvas(prediction, panelSize);
    cv::Mat pointCloudPanel = fitToCanvas(pointCloud, panelSize);

    cv::Mat output(
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        CV_8UC3,
        cv::Scalar(15, 15, 15)
    );

    originalPanel.copyTo(
        output(cv::Rect(0, 0, panelWidth, OUTPUT_HEIGHT))
    );

    predictionPanel.copyTo(
        output(cv::Rect(
            panelWidth + PANEL_GAP,
            0,
            panelWidth,
            OUTPUT_HEIGHT
        ))
    );

    pointCloudPanel.copyTo(
        output(cv::Rect(
            2 * (panelWidth + PANEL_GAP),
            0,
            panelWidth,
            OUTPUT_HEIGHT
        ))
    );

    cv::putText(
        output,
        "Camera",
        cv::Point(15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    cv::putText(
        output,
        "Inference",
        cv::Point(panelWidth + PANEL_GAP + 15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    cv::putText(
        output,
        "3D Reconstruction",
        cv::Point(2 * (panelWidth + PANEL_GAP) + 15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    return output;
}


bool isWindowClosed()
{
    try
    {
        return cv::getWindowProperty(
            WINDOW_NAME,
            cv::WND_PROP_VISIBLE
        ) == 0.0;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
}

}


int main(int argc, char* argv[])
{
    try
    {
        std::cout
            << "AdaptiveRobotPerception started!"
            << std::endl;

        std::cout
            << "Prediction: "
            << PREDICTION_FPS
            << " FPS"
            << std::endl;

        // Camera initialization.

        cv::VideoCapture cap;

        if (argc > 1)
            cap.open(argv[1]);
        else
            cap.open(0);

        if (!cap.isOpened())
        {
            throw std::runtime_error(
                "Could not open video source."
            );
        }

        // Prediction manager.

        PredictionManager predictionManager;

        predictionManager.addPrediction(
            std::make_unique<DepthEstimator>(
                "models/yolo26s-depth.onnx"
            )
        );

        predictionManager.addPrediction(
            std::make_unique<SegmentationEstimator>(
                "models/yolo26s-seg.onnx"
            )
        );

        predictionManager.addPrediction(
            std::make_unique<ObjectDetectionEstimator>(
                "models/yolo26s-detect.onnx"
            )
        );

        auto* depthEstimator =
            dynamic_cast<DepthEstimator*>(
                predictionManager.getPrediction(
                    PredictionType::DEPTH
                )
            );

        if (!depthEstimator)
        {
            throw std::runtime_error(
                "Depth estimator not registered."
            );
        }

        // Motion estimator.

        MotionEstimator motionEstimator;

        motionEstimator.setThreshold(
            MOTION_THRESHOLD
        );

        // Point cloud visualizer.

        PointCloudVisualizer pointCloudVisualizer(
            CAMERA_FX,
            CAMERA_FY,
            CAMERA_CX,
            CAMERA_CY
        );

        pointCloudVisualizer.setSamplingStep(
            POINT_CLOUD_SAMPLING_STEP
        );

        // Window initialization.

        cv::namedWindow(
            WINDOW_NAME,
            cv::WINDOW_NORMAL
        );

        cv::resizeWindow(
            WINDOW_NAME,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT
        );

        // Frame buffers.

        cv::Mat frame;

        cv::Mat cachedVisualization;
        cv::Mat cachedPointCloud;

        cv::Mat persistentForegroundMask;

        long long predictionCount = 0;

        bool running = true;

        auto nextPredictionTime = Clock::now();

        // Main processing loop.

        while (running)
        {
            if (!cap.read(frame) || frame.empty())
                break;

            // Estimate motion on every frame.

            const MotionState motionState =
                motionEstimator.estimateMotion(frame);

            // Prediction scheduling.

            const auto predictionNow = Clock::now();

            if (predictionNow >= nextPredictionTime)
            {
                nextPredictionTime =
                    predictionNow + PREDICTION_INTERVAL;

                ++predictionCount;

                // Alternate between depth and object perception.

                if (predictionCount % 2 == 1)
                {
                    predictionManager.runPrediction(
                        PredictionType::DEPTH,
                        frame
                    );

                    cachedVisualization =
                        predictionManager.getVisualization(
                            PredictionType::DEPTH
                        );
                }
                else
                {
                    std::string predictionType;

                    // Adaptive segmentation / detection based on motion.

                    if (motionState == MotionState::MOVING)
                    {
                        predictionManager.runPrediction(
                            PredictionType::SEGMENTATION,
                            frame
                        );

                        cachedVisualization =
                            predictionManager.getVisualization(
                                PredictionType::SEGMENTATION
                            );

                        predictionType = "Segmentation";
                    }
                    else
                    {
                        predictionManager.runPrediction(
                            PredictionType::OBJECT_DETECTION,
                            frame
                        );

                        cachedVisualization =
                            predictionManager.getVisualization(
                                PredictionType::OBJECT_DETECTION
                            );

                        predictionType = "Object Detection";
                    }

                    // Update foreground mask.

                    cv::Mat newMask = createForegroundMask(
                        frame,
                        cachedVisualization,
                        predictionType
                    );

                    if (!newMask.empty())
                    {
                        const int foregroundPixels =
                            cv::countNonZero(newMask);

                        if (foregroundPixels >= MIN_MASK_PIXELS)
                        {
                            persistentForegroundMask =
                                newMask.clone();
                        }
                    }
                }

                // Ensure valid visualization.

                if (cachedVisualization.empty())
                {
                    cachedVisualization = cv::Mat(
                        frame.size(),
                        CV_8UC3,
                        cv::Scalar(30, 30, 30)
                    );
                }
            }

            // Update point cloud.

            if (
                depthEstimator->hasResult() &&
                !depthEstimator->getDepth().empty() &&
                !persistentForegroundMask.empty()
            )
            {
                pointCloudVisualizer.update(
                    frame,
                    depthEstimator->getDepth(),
                    persistentForegroundMask
                );

                cachedPointCloud =
                    pointCloudVisualizer.getVisualization();
            }

            // Compose visualization.

            cv::Mat combined = createOutput(
                frame,
                cachedVisualization,
                cachedPointCloud
            );

            cv::imshow(
                WINDOW_NAME,
                combined
            );

            // Handle keyboard input.

            const int key = cv::waitKey(1);

            if (
                key == 27 ||
                key == 'q' ||
                key == 'Q'
            )
            {
                running = false;
                break;
            }

            if (isWindowClosed())
            {
                running = false;
                break;
            }
        }

        // Cleanup.

        cap.release();

        cv::destroyAllWindows();

        std::cout
            << "AdaptivePerception stopped!"
            << std::endl;
    }
    catch (const std::exception& e)
    {
        cv::destroyAllWindows();

        std::cerr
            << "Error: "
            << e.what()
            << std::endl;

        return 1;
    }

    return 0;
}