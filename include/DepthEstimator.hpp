#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>

class DepthEstimator {
public:
    explicit DepthEstimator(const std::string& modelPath);

    // Returns a CV_32F depth map in meters.
    cv::Mat estimateDepth(const cv::Mat& frame);

private:
    Ort::Env env;
    Ort::Session session;

    const int inputWidth = 768;
    const int inputHeight = 768;
};
