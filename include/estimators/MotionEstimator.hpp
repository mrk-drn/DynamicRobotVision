#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaoptflow.hpp>

// Motion classification
enum class MotionState
{
    STATIC,
    MOVING
};


class MotionEstimator
{
public:

    MotionEstimator();

    MotionState estimateMotion(const cv::Mat& frame);

    void setThreshold(double threshold);

    void reset();

    bool isCudaEnabled() const;

private:

    MotionState estimateMotionCPU(
        const cv::Mat& gray
    );

    MotionState estimateMotionCUDA(
        const cv::Mat& frame
    );

    double motionThreshold = 2.0;

    int maxCorners = 300;

    double qualityLevel = 0.01;

    double minDistance = 10.0;

    bool initialized = false;

    bool useCUDA = false;

    cv::Mat previousGray;

    // Persistent GPU buffers

    cv::cuda::GpuMat inputGPU;

    cv::cuda::GpuMat currentGrayGPU;

    cv::cuda::GpuMat previousGrayGPU;

    cv::cuda::GpuMat previousPointsGPU;

    cv::cuda::GpuMat currentPointsGPU;

    cv::cuda::GpuMat statusGPU;

    cv::cuda::GpuMat errorsGPU;

    cv::cuda::GpuMat resultGPU;

    // CUDA algorithms

    cv::Ptr<cv::cuda::CornersDetector> featureDetector;

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> opticalFlow;
};