
#pragma once

#include "core/Prediction.hpp"

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <cstddef>

class ObjectDetectionEstimator : public Prediction {
public:

    explicit ObjectDetectionEstimator(
        const std::string& modelPath
    );

    ~ObjectDetectionEstimator() override;

    ObjectDetectionEstimator(
        const ObjectDetectionEstimator&
    ) = delete;

    ObjectDetectionEstimator& operator=(
        const ObjectDetectionEstimator&
    ) = delete;

    PredictionType type() const override;

    void predict(
        const cv::Mat& frame
    ) override;

    cv::Mat visualize() const override;

    bool hasResult() const override;

    std::string name() const override;

    // Pure BGR detection colour map for point-cloud rendering.
    cv::Mat getDetectionColorMap() const;

private:

    Ort::Env env;
    Ort::Session session{nullptr};

    std::string inputName_;
    std::string outputName_;

    cv::Mat visualization_;
    cv::Mat detectionColorMap_;

    unsigned char* gpuFrame_ = nullptr;
    float* gpuInput_ = nullptr;

    std::size_t gpuFrameCapacity_ = 0;

    void ensureGpuBuffers(
        int width,
        int height
    );

    void releaseGpuBuffers() noexcept;
};
