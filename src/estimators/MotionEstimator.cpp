
#include "estimators/MotionEstimator.hpp"

MotionEstimator::MotionEstimator() = default;

MotionState MotionEstimator::estimateMotion(const cv::Mat& frame) {

    if (frame.empty()) {
        return MotionState::STATIC;
    }

    // Convert current frame to grayscale
    cv::Mat currentGray;
    cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);

    // Initialize using the first frame
    if (!initialized) {
        previousGray = currentGray.clone();
        initialized = true;
        return MotionState::STATIC;
    }

    // Detect features in previous frame
    std::vector<cv::Point2f> previousPoints;

    cv::goodFeaturesToTrack(
        previousGray,
        previousPoints,
        maxCorners,
        qualityLevel,
        minDistance
    );

    if (previousPoints.empty()) {
        previousGray = currentGray.clone();
        return MotionState::STATIC;
    }

    // Track features between frames
    std::vector<cv::Point2f> currentPoints;
    std::vector<uchar> status;
    std::vector<float> errors;

    cv::calcOpticalFlowPyrLK(
        previousGray,
        currentGray,
        previousPoints,
        currentPoints,
        status,
        errors
    );

    // Calculate average displacement
    double totalDisplacement = 0.0;
    int validPoints = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            double displacement = cv::norm(
                currentPoints[i] - previousPoints[i]
            );

            totalDisplacement += displacement;
            validPoints++;
        }
    }

    // Update previous frame
    previousGray = currentGray.clone();

    if (validPoints == 0) {
        return MotionState::STATIC;
    }

    double averageDisplacement =
        totalDisplacement / validPoints;

    return averageDisplacement > motionThreshold
        ? MotionState::MOVING
        : MotionState::STATIC;
}

void MotionEstimator::setThreshold(double threshold) {
    motionThreshold = threshold;
}

void MotionEstimator::reset() {
    previousGray.release();
    initialized = false;
}
