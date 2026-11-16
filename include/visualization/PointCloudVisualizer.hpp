
#pragma once

#include "visualization/PointCloudKernels.cuh"

#include <opencv2/opencv.hpp>

#include <memory>

class PointCloudVisualizer {
public:

    PointCloudVisualizer(
        float fx,
        float fy,
        float cx,
        float cy
    );

    ~PointCloudVisualizer();

    PointCloudVisualizer(
        const PointCloudVisualizer&
    ) = delete;

    PointCloudVisualizer& operator=(
        const PointCloudVisualizer&
    ) = delete;

    void setSamplingStep(int step);

    void update(
        const cv::Mat& colorFrame,
        const cv::Mat& depthMap,
        const cv::Mat& pointColors
    );

    cv::Mat getVisualization() const;

private:

    float fx_;
    float fy_;
    float cx_;
    float cy_;

    int samplingStep_ = 10;

    cv::Mat visualization_;

    std::unique_ptr<CudaPointCloudBackend> cudaBackend_;

    void renderPointCloud();
};
