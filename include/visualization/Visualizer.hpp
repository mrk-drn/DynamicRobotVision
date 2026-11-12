#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class Visualizer {
public:
    static cv::Mat compose(
        const cv::Mat& frame,
        const std::vector<cv::Mat>& predictions,
        const std::vector<std::string>& labels,
        double scale = 0.4
    );
};