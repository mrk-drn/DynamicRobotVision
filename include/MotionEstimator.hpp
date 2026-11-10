
#pragma once

#include <opencv2/opencv.hpp>

enum class MotionState {
    STATIC,
    MOVING
};

class MotionEstimator {
public:
    MotionEstimator();

    // Processes a frame and returns its motion state.
    MotionState estimateMotion(const cv::Mat& frame);

    // Configure motion threshold (pixels per frame).
    void setThreshold(double threshold);

    void reset();

private:
    cv::Mat previousGray;
    bool initialized = false;

    int maxCorners = 200;
    double qualityLevel = 0.01;
    double minDistance = 10.0;

    double motionThreshold = 2.0;
};
