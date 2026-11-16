
#pragma once

#include <opencv2/core.hpp>

#include <memory>

class CudaPointCloudBackend
{
public:
    static constexpr int WIDTH = 640;
    static constexpr int HEIGHT = 720;

    CudaPointCloudBackend();

    ~CudaPointCloudBackend();

    CudaPointCloudBackend(const CudaPointCloudBackend&) = delete;
    CudaPointCloudBackend& operator=(const CudaPointCloudBackend&) = delete;

    void render(
        const cv::Mat& depth,
        const cv::Mat& mask,
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
