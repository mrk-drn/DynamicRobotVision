#include "estimators/MotionEstimator.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>


extern "C"
int launchMotionReduction(
    const void* previousPoints,
    size_t previousStep,

    const void* currentPoints,
    size_t currentStep,

    const void* status,
    size_t statusStep,

    int rows,
    int cols,

    float threshold,

    int* result
);


MotionEstimator::MotionEstimator()
{
    try
    {
        const int deviceCount =
            cv::cuda::getCudaEnabledDeviceCount();

        if (deviceCount > 0)
        {
            cv::cuda::setDevice(0);

            featureDetector =
                cv::cuda::createGoodFeaturesToTrackDetector(
                    CV_8UC1,
                    maxCorners,
                    qualityLevel,
                    minDistance
                );

            opticalFlow =
                cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21),
                    3,
                    30
                );

            resultGPU.create(
                1,
                1,
                CV_32SC1
            );

            useCUDA = true;
        }
    }
    catch (const cv::Exception& e)
    {
        std::cerr
            << "[MotionEstimator] CUDA initialization failed: "
            << e.what()
            << std::endl;

        useCUDA = false;
    }

    std::cout
        << "[MotionEstimator] Backend: "
        << (useCUDA ? "CUDA" : "CPU")
        << std::endl;
}


// Main entry point

MotionState MotionEstimator::estimateMotion(
    const cv::Mat& frame
)
{
    if (frame.empty())
        return MotionState::STATIC;

    if (useCUDA)
    {
        try
        {
            return estimateMotionCUDA(frame);
        }
        catch (const cv::Exception& e)
        {
            std::cerr
                << "[MotionEstimator] CUDA failure: "
                << e.what()
                << std::endl;

            useCUDA = false;

            reset();
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "[MotionEstimator] CUDA failure: "
                << e.what()
                << std::endl;

            useCUDA = false;

            reset();
        }
    }

    cv::Mat gray;

    if (frame.channels() == 1)
        gray = frame;
    else if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    else
        throw std::runtime_error("Unsupported frame format.");

    return estimateMotionCPU(gray);
}


// CUDA implementation

MotionState MotionEstimator::estimateMotionCUDA(
    const cv::Mat& frame
)
{
    // Reuse persistent GPU buffers.

    inputGPU.upload(frame);

    // GPU grayscale conversion.

    if (frame.channels() == 1)
    {
        currentGrayGPU = inputGPU;
    }
    else if (frame.channels() == 3)
    {
        cv::cuda::cvtColor(
            inputGPU,
            currentGrayGPU,
            cv::COLOR_BGR2GRAY
        );
    }
    else if (frame.channels() == 4)
    {
        cv::cuda::cvtColor(
            inputGPU,
            currentGrayGPU,
            cv::COLOR_BGRA2GRAY
        );
    }
    else
    {
        throw std::runtime_error(
            "Unsupported frame format."
        );
    }

    // First frame initialization.

    if (!initialized)
    {
        currentGrayGPU.copyTo(previousGrayGPU);

        initialized = true;

        return MotionState::STATIC;
    }

    // Feature detection.

    featureDetector->detect(
        previousGrayGPU,
        previousPointsGPU
    );

    if (previousPointsGPU.empty())
    {
        currentGrayGPU.copyTo(previousGrayGPU);

        return MotionState::STATIC;
    }

    // Optical flow.

    opticalFlow->calc(
        previousGrayGPU,
        currentGrayGPU,
        previousPointsGPU,
        currentPointsGPU,
        statusGPU,
        errorsGPU
    );

    // Custom CUDA displacement reduction.

    if (
        currentPointsGPU.empty() ||
        statusGPU.empty() ||
        previousPointsGPU.size() != currentPointsGPU.size()
    )
    {
        currentGrayGPU.copyTo(previousGrayGPU);

        return MotionState::STATIC;
    }

    const int error = launchMotionReduction(
        previousPointsGPU.data,
        previousPointsGPU.step,

        currentPointsGPU.data,
        currentPointsGPU.step,

        statusGPU.data,
        statusGPU.step,

        previousPointsGPU.rows,
        previousPointsGPU.cols,

        static_cast<float>(motionThreshold),

        reinterpret_cast<int*>(resultGPU.data)
    );

    if (error != static_cast<int>(cudaSuccess))
    {
        throw std::runtime_error(
            "CUDA motion reduction launch failed."
        );
    }

    // Transfer ONLY one integer from GPU to CPU.

    int result = 0;

    const cudaError_t copyError = cudaMemcpy(
        &result,
        resultGPU.data,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (copyError != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(copyError)
        );
    }

    // Update previous frame.

    currentGrayGPU.copyTo(previousGrayGPU);

    return result == 1
        ? MotionState::MOVING
        : MotionState::STATIC;
}


// CPU fallback

MotionState MotionEstimator::estimateMotionCPU(
    const cv::Mat& currentGray
)
{
    if (!initialized)
    {
        previousGray = currentGray.clone();

        initialized = true;

        return MotionState::STATIC;
    }

    std::vector<cv::Point2f> previousPoints;

    cv::goodFeaturesToTrack(
        previousGray,
        previousPoints,
        maxCorners,
        qualityLevel,
        minDistance
    );

    if (previousPoints.empty())
    {
        previousGray = currentGray.clone();

        return MotionState::STATIC;
    }

    std::vector<cv::Point2f> currentPoints;

    std::vector<uchar> status;
    std::vector<float> errors;

    cv::calcOpticalFlowPyrLK(
        previousGray,
        currentGray,
        previousPoints,
        currentPoints,
        status,
        errors
    );

    double totalDisplacement = 0.0;

    int validPoints = 0;

    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            totalDisplacement += cv::norm(
                currentPoints[i] - previousPoints[i]
            );

            ++validPoints;
        }
    }

    previousGray = currentGray.clone();

    if (validPoints == 0)
        return MotionState::STATIC;

    const double averageDisplacement =
        totalDisplacement / validPoints;

    return averageDisplacement > motionThreshold
        ? MotionState::MOVING
        : MotionState::STATIC;
}


// Configuration

void MotionEstimator::setThreshold(double threshold)
{
    motionThreshold = std::max(0.0, threshold);
}


void MotionEstimator::reset()
{
    previousGray.release();

    inputGPU.release();
    currentGrayGPU.release();
    previousGrayGPU.release();

    previousPointsGPU.release();
    currentPointsGPU.release();

    statusGPU.release();
    errorsGPU.release();

    initialized = false;
}


bool MotionEstimator::isCudaEnabled() const
{
    return useCUDA;
}