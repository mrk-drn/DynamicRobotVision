
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

        auto* segmentationEstimator =
            dynamic_cast<SegmentationEstimator*>(
                predictionManager.getPrediction(
                    PredictionType::SEGMENTATION
                )
            );

        if (!segmentationEstimator)
        {
            throw std::runtime_error(
                "Segmentation estimator not registered."
            );
        }

        auto* detectionEstimator =
            dynamic_cast<ObjectDetectionEstimator*>(
                predictionManager.getPrediction(
                    PredictionType::OBJECT_DETECTION
                )
            );

        if (!detectionEstimator)
        {
            throw std::runtime_error(
                "Object detection estimator not registered."
            );
        }

        MotionEstimator motionEstimator;

        motionEstimator.setThreshold(
            MOTION_THRESHOLD
        );

        PointCloudVisualizer pointCloudVisualizer(
            CAMERA_FX,
            CAMERA_FY,
            CAMERA_CX,
            CAMERA_CY
        );

        pointCloudVisualizer.setSamplingStep(
            POINT_CLOUD_SAMPLING_STEP
        );

        cv::namedWindow(
            WINDOW_NAME,
            cv::WINDOW_NORMAL
        );

        cv::resizeWindow(
            WINDOW_NAME,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT
        );

        cv::Mat frame;

        cv::Mat cachedVisualization;
        cv::Mat cachedPointCloud;

        // This is the active point-cloud colour map.
        // It changes only after segmentation or detection.
        // Depth predictions deliberately leave it unchanged.

        cv::Mat cachedPointColors;

        long long predictionCount = 0;

        bool running = true;

        auto nextPredictionTime = Clock::now();

        while (running)
        {
            if (!cap.read(frame) || frame.empty())
                break;

            const MotionState motionState =
                motionEstimator.estimateMotion(frame);

            const auto predictionNow = Clock::now();

            if (predictionNow >= nextPredictionTime)
            {
                nextPredictionTime =
                    predictionNow + PREDICTION_INTERVAL;

                ++predictionCount;

                // Alternate between depth and perception.

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

                    // IMPORTANT:
                    // Do not modify cachedPointColors here.
                    // Keep the previous segmentation/detection colours.
                }
                else
                {
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

                        cachedPointColors =
                            segmentationEstimator->getSegmentationColorMap();
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

                        cachedPointColors =
                            detectionEstimator->getDetectionColorMap();
                    }
                }

                if (cachedVisualization.empty())
                {
                    cachedVisualization = cv::Mat(
                        frame.size(),
                        CV_8UC3,
                        cv::Scalar(30, 30, 30)
                    );
                }
            }

            // Reconstruct using the latest depth and the latest
            // available perception colour map.

            if (
                depthEstimator->hasResult() &&
                !depthEstimator->getDepth().empty() &&
                !cachedPointColors.empty()
            )
            {
                pointCloudVisualizer.update(
                    frame,
                    depthEstimator->getDepth(),
                    cachedPointColors
                );

                cachedPointCloud =
                    pointCloudVisualizer.getVisualization();
            }

            cv::Mat combined = createOutput(
                frame,
                cachedVisualization,
                cachedPointCloud
            );

            cv::imshow(
                WINDOW_NAME,
                combined
            );

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
