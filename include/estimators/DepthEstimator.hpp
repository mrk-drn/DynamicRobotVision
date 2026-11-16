#pragma once

#include "core/Prediction.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <cstdint>

class DepthEstimator : public Prediction {
public:
    explicit DepthEstimator(const std::string& modelPath);

    PredictionType type() const override {
        return PredictionType::DEPTH;
    }

    void predict(const cv::Mat& frame) override;

    cv::Mat visualize() const override;

    bool hasResult() const override;

    std::string name() const override {
        return "Depth Estimation";
    }

    const cv::Mat& getDepth() const;

    float nearestDistanceMeters() const;

private:
    static constexpr int INPUT_WIDTH = 768;
    static constexpr int INPUT_HEIGHT = 768;

    static constexpr size_t INPUT_ELEMENTS =
        3ULL * INPUT_WIDTH * INPUT_HEIGHT;

    Ort::Env env;
    Ort::Session session{nullptr};

    std::string inputName_;
    std::string outputName_;

    Ort::MemoryInfo memoryInfo_{
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        )
    };

    // Reusable CPU preprocessing buffers.
    cv::Mat resized_;
    cv::Mat letterboxed_;
    cv::Mat blob_;

    // Persistent results.
    cv::Mat depthResult_;
    cv::Mat depthVisualization_;

    // Cached distance calculation.
    mutable float nearestDistanceCache_ = -1.0f;
    mutable bool nearestDistanceValid_ = false;

    // Cached input shape.
    const int64_t inputShape_[4] = {
        1,
        3,
        INPUT_HEIGHT,
        INPUT_WIDTH
    };
};
