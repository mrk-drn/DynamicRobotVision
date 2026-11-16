
#pragma once

#include <opencv2/core.hpp>

#include <memory>

class CudaPointCloudBackend;

class PointCloudVisualizer
{
public:
    PointCloudVisualizer(
        float fx,
        float fy,
        float cx,
        float cy
    );

    ~PointCloudVisualizer();

    PointCloudVisualizer(const PointCloudVisualizer&) = delete;
    PointCloudVisualizer& operator=(const PointCloudVisualizer&) = delete;

    void setSamplingStep(int step);

    void update(
        const cv::Mat& colorFrame,
        const cv::Mat& depthMap,
        const cv::Mat& segmentationMask = cv::Mat()
    );

    cv::Mat getVisualization() const;

private:
    void renderPointCloud();

    float fx_;
    float fy_;
    float cx_;
    float cy_;

    int samplingStep_ = 5;

    cv::Mat visualization_;

    std::unique_ptr<CudaPointCloudBackend> cudaBackend_;
};
