#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace SegmentationKernels {

class Processor {
public:

    Processor();

    ~Processor();

    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;

    Processor(Processor&&) noexcept;
    Processor& operator=(Processor&&) noexcept;

    void process(
        const cv::Mat& frame,
        const float* prototypes,
        const cv::Mat& coefficients,
        const std::vector<cv::Vec4i>& boxes,
        const std::vector<cv::Vec3f>& colors,
        int resizedWidth,
        int resizedHeight,
        int padLeft,
        int padTop,
        float maskThreshold,
        float opacity,
        cv::Mat& visualization,
        cv::Mat& combinedMask,
        cv::Mat& colorMap
    );

private:

    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}