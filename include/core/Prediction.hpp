#pragma once

#include <opencv2/opencv.hpp>
#include <string>

enum class PredictionType {
    DEPTH,
    OBJECT_DETECTION,
    SEGMENTATION
};

class Prediction {
public:
    virtual ~Prediction() = default;

    virtual PredictionType type() const = 0;
    virtual void predict(const cv::Mat& frame) = 0;
    virtual cv::Mat visualize() const = 0;
    virtual bool hasResult() const = 0;
    virtual std::string name() const = 0;
};