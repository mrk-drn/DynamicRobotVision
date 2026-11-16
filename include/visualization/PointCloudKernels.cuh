
#pragma once

#include <opencv2/opencv.hpp>

#include <memory>

class CudaPointCloudBackend {
public:

    CudaPointCloudBackend();

    ~CudaPointCloudBackend();

    CudaPointCloudBackend(
        const CudaPointCloudBackend&
    ) = delete;

    CudaPointCloudBackend& operator=(
        const CudaPointCloudBackend&
    ) = delete;

    void render(
        const cv::Mat& depth,
        const cv::Mat& pointColors,
        int samplingStep,
        float fx,
        float fy,
        float cx,
        float cy,
        cv::Mat& output
    );

private:

    struct Impl;

    std::unique_ptr<Impl> impl_;
};
