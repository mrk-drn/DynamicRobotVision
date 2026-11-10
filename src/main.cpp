#include <iostream>
#include <opencv2/opencv.hpp>

#include "MotionEstimator.hpp"
#include "DepthEstimator.hpp"

// Convert metric depth into a color visualization.
cv::Mat colorizeDepth(const cv::Mat& depth) {

    cv::Mat normalizedDepth;

    // Normalize depth values to [0, 255]
    cv::normalize(
        depth,
        normalizedDepth,
        0,
        255,
        cv::NORM_MINMAX,
        CV_8U
    );

    // Convert to grayscale visualization
    return normalizedDepth;
}

int main(int argc, char* argv[]) {

    try {

        std::cout << "AdaptivePerception started!" << std::endl;

        cv::VideoCapture cap;

        // Open video file or webcam
        if (argc > 1) {

            std::string videoPath = argv[1];

            std::cout << "Loading video: "
                      << videoPath << std::endl;

            cap.open(videoPath);

            if (!cap.isOpened()) {
                std::cerr << "Could not open video file: "
                          << videoPath << std::endl;
                return 1;
            }
        }
        else {

            std::cout << "No video provided. Trying webcam..."
                      << std::endl;

            cap.open(0);

            if (!cap.isOpened()) {
                std::cout << "No webcam found. Exiting."
                          << std::endl;
                return 0;
            }
        }

        // Initialize motion estimator
        MotionEstimator motionEstimator;
        motionEstimator.setThreshold(10.0);

        // Initialize GPU depth estimator
        DepthEstimator depthEstimator(
            "models/yolo26n-depth.onnx"
        );

        cv::Mat frame;

        // Store latest depth prediction
        cv::Mat lastDepth;
        bool hasDepth = false;

        while (true) {

            if (!cap.read(frame) || frame.empty()) {
                std::cout << "End of video or failed to capture frame."
                          << std::endl;
                break;
            }

            // Estimate motion
            MotionState motionState =
                motionEstimator.estimateMotion(frame);

            // Run depth estimation when motion is detected
            if (motionState == MotionState::MOVING) {

                lastDepth = depthEstimator.estimateDepth(frame);

                hasDepth = !lastDepth.empty();
            }

            // Create visualization
            cv::Mat visualization;

            if (hasDepth) {

                visualization = colorizeDepth(lastDepth);

                std::string label =
                    motionState == MotionState::MOVING
                    ? "Depth Estimation (GPU)"
                    : "Depth (Cached)";

                cv::putText(
                    visualization,
                    label,
                    cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }
            else {

                visualization = cv::Mat(
                    frame.size(),
                    CV_8UC3,
                    cv::Scalar(30, 30, 30)
                );

                cv::putText(
                    visualization,
                    "Waiting for motion...",
                    cv::Point(20, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }

            // Label original frame
            cv::putText(
                frame,
                "Original Frame",
                cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2
            );

            // Resize both images
            cv::Mat resizedFrame;
            cv::Mat resizedVisualization;

            // Convert grayscale depth visualization to BGR
            if (visualization.channels() == 1) {
                cv::cvtColor(
                    visualization,
                    visualization,
                    cv::COLOR_GRAY2BGR
                );
            }

            cv::resize(
                frame,
                resizedFrame,
                cv::Size(),
                0.4,
                0.4
            );

            cv::resize(
                visualization,
                resizedVisualization,
                cv::Size(),
                0.4,
                0.4
            );

            // Combine images
            cv::Mat combined;

            cv::hconcat(
                resizedFrame,
                resizedVisualization,
                combined
            );

            cv::imshow("AdaptivePerception", combined);

            if (cv::waitKey(1) == 27) {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

        std::cout << "AdaptivePerception stopped!" << std::endl;

    }
    catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
