#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "core/PredictionManager.hpp"
#include "estimators/MotionEstimator.hpp"
#include "estimators/DepthEstimator.hpp"
#include "estimators/SegmentationEstimator.hpp"
#include "visualization/Visualizer.hpp"

namespace {

constexpr double TARGET_FPS = 15.0;

constexpr int OUTPUT_WIDTH = 1280;
constexpr int OUTPUT_HEIGHT = 720;

constexpr float CLOSE_DISTANCE_THRESHOLD = 3.0f;

const std::string WINDOW_NAME = "AdaptivePerception";

cv::Mat fitToCanvas(
    const cv::Mat& source,
    const cv::Size& canvasSize,
    const cv::Scalar& background = cv::Scalar(25, 25, 25)
) {
    cv::Mat canvas(
        canvasSize,
        CV_8UC3,
        background
    );

    if (source.empty()) {
        return canvas;
    }

    const double scale = std::min(
        static_cast<double>(canvasSize.width) / source.cols,
        static_cast<double>(canvasSize.height) / source.rows
    );

    const int width = std::max(
        1,
        static_cast<int>(std::round(source.cols * scale))
    );

    const int height = std::max(
        1,
        static_cast<int>(std::round(source.rows * scale))
    );

    cv::Mat resized;

    cv::resize(
        source,
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
    const std::string& label
) {
    constexpr int PANEL_GAP = 4;

    const int panelWidth =
        (OUTPUT_WIDTH - PANEL_GAP) / 2;

    cv::Mat originalPanel = fitToCanvas(
        frame,
        cv::Size(panelWidth, OUTPUT_HEIGHT)
    );

    cv::Mat predictionPanel = fitToCanvas(
        prediction,
        cv::Size(panelWidth, OUTPUT_HEIGHT)
    );

    cv::Mat output(
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        CV_8UC3,
        cv::Scalar(15, 15, 15)
    );

    originalPanel.copyTo(
        output(cv::Rect(
            0,
            0,
            panelWidth,
            OUTPUT_HEIGHT
        ))
    );

    predictionPanel.copyTo(
        output(cv::Rect(
            panelWidth + PANEL_GAP,
            0,
            panelWidth,
            OUTPUT_HEIGHT
        ))
    );

    cv::putText(
        output,
        "Original",
        cv::Point(15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    cv::putText(
        output,
        label,
        cv::Point(panelWidth + PANEL_GAP + 15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    return output;
}

// Check whether the GTK2 window still exists.
bool isWindowClosed() {
    try {
        const double property = cv::getWindowProperty(
            WINDOW_NAME,
            cv::WND_PROP_AUTOSIZE
        );

        return property < 0.0;
    }
    catch (const cv::Exception&) {
        return true;
    }
}

}

int main(int argc, char* argv[]) {

    try {

        std::cout << "AdaptivePerception started!"
                  << std::endl;

        cv::VideoCapture cap;

        if (argc > 1) {

            const std::string videoPath = argv[1];

            std::cout << "Loading video: "
                      << videoPath << std::endl;

            cap.open(videoPath);
        }
        else {

            std::cout << "No video provided. Trying webcam..."
                      << std::endl;

            cap.open(0);
        }

        if (!cap.isOpened()) {
            throw std::runtime_error(
                "Could not open video source."
            );
        }

        double videoFPS = cap.get(cv::CAP_PROP_FPS);

        if (!std::isfinite(videoFPS) || videoFPS <= 0.0) {
            videoFPS = TARGET_FPS;

            std::cout
                << "Warning: Could not determine video FPS. "
                << "Assuming 15 FPS."
                << std::endl;
        }

        const double framesPerPrediction =
            videoFPS / TARGET_FPS;

        std::cout << "Video FPS: "
                  << videoFPS << std::endl;

        std::cout << "Target prediction FPS: "
                  << TARGET_FPS << std::endl;

        std::cout << "Average frames per prediction: "
                  << framesPerPrediction << std::endl;

        MotionEstimator motionEstimator;
        motionEstimator.setThreshold(15.0);

        PredictionManager predictionManager;

        predictionManager.addPrediction(
            std::make_unique<DepthEstimator>(
                "models/yolo26n-depth.onnx"
            )
        );

        predictionManager.addPrediction(
            std::make_unique<SegmentationEstimator>(
                "models/yolo26n-seg.onnx"
            )
        );

        auto* depthEstimator =
            dynamic_cast<DepthEstimator*>(
                predictionManager.getPrediction(
                    PredictionType::DEPTH
                )
            );

        if (!depthEstimator) {
            throw std::runtime_error(
                "Depth estimator not registered."
            );
        }

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

        std::string cachedLabel = "Waiting for prediction...";

        long long frameIndex = 0;
        double nextPredictionFrame = 0.0;

        bool running = true;

        while (running) {

            if (!cap.read(frame) || frame.empty()) {
                break;
            }

            const bool runPrediction =
                static_cast<double>(frameIndex) + 1e-9 >=
                nextPredictionFrame;

            if (runPrediction) {

                nextPredictionFrame += framesPerPrediction;

                const MotionState motionState =
                    motionEstimator.estimateMotion(frame);

                bool showSegmentation = false;

                if (motionState == MotionState::MOVING) {

                    predictionManager.runPrediction(
                        PredictionType::DEPTH,
                        frame
                    );
                }
                else if (depthEstimator->hasResult()) {

                    const float nearestDistance =
                        depthEstimator->nearestDistanceMeters();

                    if (
                        nearestDistance >= 0.0f &&
                        nearestDistance < CLOSE_DISTANCE_THRESHOLD
                    ) {

                        predictionManager.runPrediction(
                            PredictionType::SEGMENTATION,
                            frame
                        );

                        showSegmentation = true;
                    }
                }

                if (showSegmentation) {

                    cachedVisualization =
                        predictionManager.getVisualization(
                            PredictionType::SEGMENTATION
                        );

                    cachedLabel = "Segmentation";
                }
                else {

                    cachedVisualization =
                        predictionManager.getVisualization(
                            PredictionType::DEPTH
                        );

                    cachedLabel =
                        motionState == MotionState::MOVING
                        ? "Depth"
                        : "Depth (Cached)";
                }

                if (cachedVisualization.empty()) {

                    cachedVisualization = cv::Mat(
                        frame.size(),
                        CV_8UC3,
                        cv::Scalar(30, 30, 30)
                    );

                    cachedLabel = "Waiting for prediction...";
                }
            }

            cv::Mat combined = createOutput(
                frame,
                cachedVisualization,
                cachedLabel
            );

            cv::imshow(
                WINDOW_NAME,
                combined
            );

            // Process GTK2 window events.
            const int key = cv::waitKey(1);

            // ESC closes the application.
            if (key == 27) {
                running = false;
                break;
            }

            // Check for the X-button close event.
            // This check must happen before the next imshow().
            if (isWindowClosed()) {

                std::cout << "Window closed by user."
                          << std::endl;

                running = false;
                break;
            }

            ++frameIndex;
        }

        cap.release();

        cv::destroyAllWindows();

        std::cout << "AdaptivePerception stopped!"
                  << std::endl;

    }
    catch (const std::exception& e) {

        cv::destroyAllWindows();

        std::cerr << "Error: "
                  << e.what()
                  << std::endl;

        return 1;
    }

    return 0;
}
