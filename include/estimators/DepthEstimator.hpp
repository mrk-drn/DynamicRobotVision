#pragma once

#include "core/Prediction.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>

class DepthEstimator : public Prediction {
public:
    explicit DepthEstimator(const std::string& modelPath);

    PredictionType type() const override {
        return PredictionType::DEPTH;
    }

    void predict(const cv::Mat& frame) override;

    std::string inputName_;
    std::string outputName_;

    cv::Mat depthVisualization_;
    cv::Mat visualize() const override;

    bool hasResult() const override;

    std::string name() const override {
        return "Depth Estimation";
    }

    const cv::Mat& getDepth() const;

    float nearestDistanceMeters() const;

private:
    Ort::Env env;
    Ort::Session session;

    const int inputWidth = 768; //640;
    const int inputHeight = 768; //640;

    cv::Mat depthResult_;
};