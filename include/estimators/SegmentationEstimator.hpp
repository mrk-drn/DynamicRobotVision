
#pragma once

#include "core/Prediction.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>

class SegmentationEstimator : public Prediction {
public:
    explicit SegmentationEstimator(const std::string& modelPath);

    PredictionType type() const override;
    void predict(const cv::Mat& frame) override;
    cv::Mat visualize() const override;
    bool hasResult() const override;
    std::string name() const override;

private:
    static constexpr int inputWidth = 640;
    static constexpr int inputHeight = 640;

    Ort::Env env;
    Ort::Session session;

    cv::Mat visualization_;
};
