
#pragma once

#include <opencv2/opencv.hpp>

#include <vtkSmartPointer.h>

class vtkRenderer;
class vtkRenderWindow;
class vtkPolyData;
class vtkPolyDataMapper;
class vtkActor;
class vtkWindowToImageFilter;

class PointCloudVisualizer {
public:
    PointCloudVisualizer(
        float fx,
        float fy,
        float cx,
        float cy
    );

    ~PointCloudVisualizer();

    void setSamplingStep(int step);

    void update(
        const cv::Mat& colorFrame,
        const cv::Mat& depthMap,
        const cv::Mat& segmentationMask = cv::Mat()
    );

    cv::Mat getVisualization() const;

private:
    void initializeRenderer();

    void generatePointCloud(
        const cv::Mat& colorFrame,
        const cv::Mat& depthMap,
        const cv::Mat& segmentationMask
    );

    void renderPointCloud();

    float fx_;
    float fy_;
    float cx_;
    float cy_;

    int samplingStep_ = 10;

    cv::Mat visualization_;

    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkRenderWindow> renderWindow_;
    vtkSmartPointer<vtkPolyData> polyData_;
    vtkSmartPointer<vtkPolyDataMapper> mapper_;
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkWindowToImageFilter> capture_;
};
