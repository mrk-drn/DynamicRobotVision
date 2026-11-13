#pragma once

#include "core/Prediction.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>

class ObjectDetectionEstimator : public Prediction {
public:
    explicit ObjectDetectionEstimator(
        const std::string& modelPath
    );

    PredictionType type() const override;
    void predict(const cv::Mat& frame) override;
    cv::Mat visualize() const override;
    bool hasResult() const override;
    std::string name() const override;

private:
    static constexpr int inputWidth = 640;
    static constexpr int inputHeight = 640;

    Ort::Env env;
    Ort::Session session{nullptr};

    std::string inputName_;
    std::string outputName_;

    cv::Mat visualization_;
};