
#include "visualization/PointCloudVisualizer.hpp"
#include "visualization/PointCloudKernels.cuh"
#include "visualization/PointCloudConfig.cuh"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace
{

// Colors

const cv::Scalar CAMERA_COLOR(80, 255, 80);

const cv::Scalar BACKGROUND_COLOR(
    PointCloudConfig::BACKGROUND_BLUE,
    PointCloudConfig::BACKGROUND_GREEN,
    PointCloudConfig::BACKGROUND_RED
);

const cv::Scalar TITLE_COLOR(220, 220, 235);
const cv::Scalar INFO_COLOR(150, 160, 180);

// Point representation

struct Point3D
{
    float x;
    float y;
    float z;
};

// Projection

bool projectPoint(
    const Point3D& p,
    cv::Point2f& screen,
    float& cameraDepth
)
{
    static const float cosYaw =
        std::cos(PointCloudConfig::CAMERA_YAW);

    static const float sinYaw =
        std::sin(PointCloudConfig::CAMERA_YAW);

    static const float cosPitch =
        std::cos(PointCloudConfig::CAMERA_PITCH);

    static const float sinPitch =
        std::sin(PointCloudConfig::CAMERA_PITCH);

    const float rotatedX =
        p.x * cosYaw - p.y * sinYaw;

    const float rotatedY =
        p.x * sinYaw + p.y * cosYaw;

    const float rotatedZ =
        rotatedY * sinPitch + p.z * cosPitch;

    cameraDepth =
        rotatedY * cosPitch - p.z * sinPitch;

    const float denominator =
        PointCloudConfig::CAMERA_FOCAL_LENGTH +
        (cameraDepth - PointCloudConfig::CAMERA_DISTANCE) *
        PointCloudConfig::PROJECTION_SCALE;

    if (
        !std::isfinite(denominator) ||
        denominator <= 1.0f
    )
    {
        return false;
    }

    const float perspective =
        PointCloudConfig::CAMERA_FOCAL_LENGTH /
        denominator;

    const float screenX =
        PointCloudConfig::RENDER_WIDTH * 0.5f +
        rotatedX *
        PointCloudConfig::PROJECTION_SCALE *
        perspective;

    const float screenY =
        PointCloudConfig::RENDER_HEIGHT * 0.5f -
        rotatedZ *
        PointCloudConfig::PROJECTION_SCALE *
        perspective;

    if (
        !std::isfinite(screenX) ||
        !std::isfinite(screenY)
    )
    {
        return false;
    }

    screen = cv::Point2f(screenX, screenY);

    return true;
}

// Camera frustum

void drawFrustum(cv::Mat& image)
{
    const Point3D origin = {
        0.0f,
        0.0f,
        0.0f
    };

    const Point3D corners3D[] = {
        {
            -PointCloudConfig::FRUSTUM_HALF_WIDTH,
            PointCloudConfig::FRUSTUM_FAR_DISTANCE,
            -PointCloudConfig::FRUSTUM_HALF_HEIGHT
        },
        {
            PointCloudConfig::FRUSTUM_HALF_WIDTH,
            PointCloudConfig::FRUSTUM_FAR_DISTANCE,
            -PointCloudConfig::FRUSTUM_HALF_HEIGHT
        },
        {
            PointCloudConfig::FRUSTUM_HALF_WIDTH,
            PointCloudConfig::FRUSTUM_FAR_DISTANCE,
            PointCloudConfig::FRUSTUM_HALF_HEIGHT
        },
        {
            -PointCloudConfig::FRUSTUM_HALF_WIDTH,
            PointCloudConfig::FRUSTUM_FAR_DISTANCE,
            PointCloudConfig::FRUSTUM_HALF_HEIGHT
        }
    };

    cv::Point2f cameraScreen;
    float cameraDepth = 0.0f;

    if (!projectPoint(
        origin,
        cameraScreen,
        cameraDepth
    ))
    {
        return;
    }

    cv::Point2f corners[4];

    for (int i = 0; i < 4; ++i)
    {
        if (!projectPoint(
            corners3D[i],
            corners[i],
            cameraDepth
        ))
        {
            return;
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        cv::line(
            image,
            cameraScreen,
            corners[i],
            CAMERA_COLOR,
            PointCloudConfig::FRUSTUM_LINE_THICKNESS,
            cv::LINE_AA
        );

        cv::line(
            image,
            corners[i],
            corners[(i + 1) % 4],
            CAMERA_COLOR,
            PointCloudConfig::FRUSTUM_LINE_THICKNESS,
            cv::LINE_AA
        );
    }
}

} // namespace

// Constructor

PointCloudVisualizer::PointCloudVisualizer(
    float fx,
    float fy,
    float cx,
    float cy
)
    : fx_(fx),
      fy_(fy),
      cx_(cx),
      cy_(cy),
      cudaBackend_(
          std::make_unique<CudaPointCloudBackend>()
      )
{
    if (
        !std::isfinite(fx_) ||
        !std::isfinite(fy_) ||
        fx_ <= 0.0f ||
        fy_ <= 0.0f
    )
    {
        throw std::invalid_argument(
            "Invalid camera focal lengths."
        );
    }

    visualization_ = cv::Mat(
        PointCloudConfig::RENDER_HEIGHT,
        PointCloudConfig::RENDER_WIDTH,
        CV_8UC3,
        BACKGROUND_COLOR
    );
}

// Destructor

PointCloudVisualizer::~PointCloudVisualizer() = default;

// Sampling configuration

void PointCloudVisualizer::setSamplingStep(int step)
{
    if (step < 1)
    {
        throw std::invalid_argument(
            "Sampling step must be at least 1."
        );
    }

    samplingStep_ = step;
}

// Main update

void PointCloudVisualizer::update(
    const cv::Mat& colorFrame,
    const cv::Mat& depthMap,
    const cv::Mat& segmentationMask
)
{
    (void)colorFrame;

    if (depthMap.empty())
    {
        visualization_.setTo(BACKGROUND_COLOR);
        renderPointCloud();
        return;
    }

    cv::Mat depth;

    if (depthMap.type() == CV_32FC1)
    {
        depth = depthMap;
    }
    else
    {
        depthMap.convertTo(depth, CV_32FC1);
    }

    cv::Mat mask;

    if (!segmentationMask.empty())
    {
        if (segmentationMask.channels() == 1)
        {
            mask = segmentationMask;
        }
        else if (segmentationMask.channels() == 3)
        {
            cv::cvtColor(
                segmentationMask,
                mask,
                cv::COLOR_BGR2GRAY
            );
        }
        else if (segmentationMask.channels() == 4)
        {
            cv::cvtColor(
                segmentationMask,
                mask,
                cv::COLOR_BGRA2GRAY
            );
        }
        else
        {
            throw std::runtime_error(
                "Unsupported segmentation mask channels."
            );
        }

        if (mask.size() != depth.size())
        {
            cv::resize(
                mask,
                mask,
                depth.size(),
                0,
                0,
                cv::INTER_NEAREST
            );
        }

        if (mask.type() != CV_8UC1)
        {
            mask.convertTo(mask, CV_8UC1);
        }
    }

    const float scaleX =
        static_cast<float>(depth.cols) / 1280.0f;

    const float scaleY =
        static_cast<float>(depth.rows) / 720.0f;

    const float fx = fx_ * scaleX;
    const float fy = fy_ * scaleY;

    const float cx = cx_ * scaleX;
    const float cy = cy_ * scaleY;

    if (
        !std::isfinite(fx) ||
        !std::isfinite(fy) ||
        fx <= 0.0f ||
        fy <= 0.0f
    )
    {
        visualization_.setTo(BACKGROUND_COLOR);
        renderPointCloud();
        return;
    }

    cudaBackend_->render(
        depth,
        mask,
        samplingStep_,
        fx,
        fy,
        cx,
        cy,
        visualization_
    );

    renderPointCloud();
}

// Interface rendering

void PointCloudVisualizer::renderPointCloud()
{
    drawFrustum(visualization_);
}

// Visualization access

cv::Mat PointCloudVisualizer::getVisualization() const
{
    return visualization_.clone();
}
