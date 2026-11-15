
#include "visualization/PointCloudVisualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int RENDER_WIDTH = 640;
constexpr int RENDER_HEIGHT = 720;
constexpr int DEFAULT_SAMPLING_STEP = 5;

const cv::Scalar STANDARD_POINT_COLOR(255, 200, 80);

constexpr float CAMERA_FOCAL_LENGTH = 650.0f;
constexpr float CAMERA_YAW = -0.05f;
constexpr float CAMERA_PITCH = 0.10f;

// Top-down view:
// constexpr float CAMERA_YAW = 0.0f;
// constexpr float CAMERA_PITCH = 1.5707963f;

constexpr int POINT_RADIUS = 1.0;

// Fixed world-to-screen scale
constexpr float PROJECTION_SCALE = 0.30f;

constexpr float FRUSTUM_FAR_DISTANCE = 2000.0f;
constexpr float FRUSTUM_HALF_WIDTH = 750.0f;
constexpr float FRUSTUM_HALF_HEIGHT = 450.0f;

constexpr int FRUSTUM_LINE_THICKNESS = 4;

const cv::Scalar CAMERA_COLOR(80, 255, 80);

struct Point3D {
    float x;
    float y;
    float z;
};

struct ProjectedPoint {
    cv::Point2f position;
    float depth;
};

std::vector<Point3D> currentPoints;

void savePointCloudPLY(
    const std::string& filename,
    const std::vector<Point3D>& points
)
{
    if (points.empty()) {
        std::cerr << "No points to save.\n";
        return;
    }

    std::filesystem::create_directories("output");

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not save point cloud: "
                  << filename << "\n";
        return;
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    file << std::fixed << std::setprecision(6);

    constexpr int red = 80;
    constexpr int green = 200;
    constexpr int blue = 255;

    for (const auto& point : points) {
        file
            << point.x << " "
            << point.y << " "
            << point.z << " "
            << red << " "
            << green << " "
            << blue << "\n";
    }

    file.close();

    std::cout << "Saved point cloud: "
              << filename << "\n"
              << "Points: " << points.size() << "\n";
}

}

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
      samplingStep_(DEFAULT_SAMPLING_STEP)
{
    if (
        !std::isfinite(fx_) ||
        !std::isfinite(fy_) ||
        fx_ <= 0.0f ||
        fy_ <= 0.0f
    ) {
        throw std::invalid_argument(
            "Invalid camera focal lengths."
        );
    }

    visualization_ = cv::Mat(
        RENDER_HEIGHT,
        RENDER_WIDTH,
        CV_8UC3,
        cv::Scalar(15, 15, 25)
    );
}

PointCloudVisualizer::~PointCloudVisualizer() = default;

void PointCloudVisualizer::initializeRenderer()
{
}

void PointCloudVisualizer::setSamplingStep(int step)
{
    if (step < 1) {
        throw std::invalid_argument(
            "Sampling step must be at least 1."
        );
    }

    samplingStep_ = step;
}

void PointCloudVisualizer::generatePointCloud(
    const cv::Mat& colorFrame,
    const cv::Mat& depthMap,
    const cv::Mat& segmentationMask
)
{
    (void)colorFrame;

    if (depthMap.empty()) {
        currentPoints.clear();
        return;
    }

    cv::Mat depth;

    if (depthMap.type() == CV_32FC1) {
        depth = depthMap;
    }
    else {
        depthMap.convertTo(depth, CV_32FC1);
    }

    cv::Mat mask;

    if (!segmentationMask.empty()) {
        if (segmentationMask.channels() == 1) {
            mask = segmentationMask.clone();
        }
        else if (segmentationMask.channels() == 3) {
            cv::cvtColor(
                segmentationMask,
                mask,
                cv::COLOR_BGR2GRAY
            );
        }
        else if (segmentationMask.channels() == 4) {
            cv::cvtColor(
                segmentationMask,
                mask,
                cv::COLOR_BGRA2GRAY
            );
        }
        else {
            throw std::runtime_error(
                "Unsupported segmentation mask channels."
            );
        }

        if (mask.size() != depth.size()) {
            cv::resize(
                mask,
                mask,
                depth.size(),
                0,
                0,
                cv::INTER_NEAREST
            );
        }

        if (mask.type() != CV_8UC1) {
            mask.convertTo(mask, CV_8UC1);
        }
    }

    const int width = depth.cols;
    const int height = depth.rows;

    const float scaleX =
        static_cast<float>(width) / 1280.0f;

    const float scaleY =
        static_cast<float>(height) / 720.0f;

    const float fx = fx_ * scaleX;
    const float fy = fy_ * scaleY;
    const float cx = cx_ * scaleX;
    const float cy = cy_ * scaleY;

    if (fx <= 0.0f || fy <= 0.0f) {
        currentPoints.clear();
        return;
    }

    std::vector<Point3D> points;

    points.reserve(
        static_cast<size_t>(width / samplingStep_) *
        static_cast<size_t>(height / samplingStep_)
    );

    for (int v = 0; v < height; v += samplingStep_) {
        for (int u = 0; u < width; u += samplingStep_) {

            if (
                !mask.empty() &&
                mask.at<unsigned char>(v, u) == 0
            ) {
                continue;
            }

            const float rawDepth =
                depth.at<float>(v, u);

            if (
                !std::isfinite(rawDepth) ||
                rawDepth <= 0.0f
            ) {
                continue;
            }

            const float distance = rawDepth * 1000.0f;

            const float x =
                (static_cast<float>(u) - cx) *
                distance / fx;

            const float y = distance;

            const float z =
                -(static_cast<float>(v) - cy) *
                distance / fy;

            if (
                !std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z)
            ) {
                continue;
            }

            points.push_back({
                x,
                y,
                z
            });
        }
    }

    static bool firstCloudSaved = false;

    if (!firstCloudSaved && !points.empty()) {
        savePointCloudPLY(
            "output/pointcloud_first_frame.ply",
            points
        );

        firstCloudSaved = true;
    }

    currentPoints = std::move(points);

    if (currentPoints.empty()) {
        return;
    }

    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();

    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();

    for (const auto& p : currentPoints) {
        minX = std::min(minX, static_cast<double>(p.x));
        maxX = std::max(maxX, static_cast<double>(p.x));

        minY = std::min(minY, static_cast<double>(p.y));
        maxY = std::max(maxY, static_cast<double>(p.y));

        minZ = std::min(minZ, static_cast<double>(p.z));
        maxZ = std::max(maxZ, static_cast<double>(p.z));
    }

}


void PointCloudVisualizer::renderPointCloud()
{
    visualization_ = cv::Mat(
        RENDER_HEIGHT,
        RENDER_WIDTH,
        CV_8UC3,
        cv::Scalar(15, 15, 25)
    );

    const float yaw = CAMERA_YAW;
    const float pitch = CAMERA_PITCH;

    const float cosYaw = std::cos(yaw);
    const float sinYaw = std::sin(yaw);

    const float cosPitch = std::cos(pitch);
    const float sinPitch = std::sin(pitch);

    // Fixed projection relative to the camera origin.
    auto projectPoint =
        [&](const Point3D& p,
            cv::Point2f& screen,
            float& cameraDepth) -> bool {

        const float rotatedX =
            p.x * cosYaw - p.y * sinYaw;

        const float rotatedY =
            p.x * sinYaw + p.y * cosYaw;

        const float rotatedZ =
            rotatedY * sinPitch + p.z * cosPitch;

        cameraDepth =
            rotatedY * cosPitch - p.z * sinPitch;

        const float denominator =
            CAMERA_FOCAL_LENGTH +
            cameraDepth * PROJECTION_SCALE;

        if (
            !std::isfinite(denominator) ||
            denominator <= 1.0f
        ) {
            return false;
        }

        const float perspective =
            CAMERA_FOCAL_LENGTH / denominator;

        const float screenX =
            RENDER_WIDTH * 0.5f +
            rotatedX * PROJECTION_SCALE * perspective;

        const float screenY =
            RENDER_HEIGHT * 0.5f -
            rotatedZ * PROJECTION_SCALE * perspective;

        if (
            !std::isfinite(screenX) ||
            !std::isfinite(screenY)
        ) {
            return false;
        }

        screen = cv::Point2f(screenX, screenY);

        return true;
    };

    std::vector<ProjectedPoint> projected;

    projected.reserve(currentPoints.size());

    for (const auto& p : currentPoints) {

        cv::Point2f screen;
        float pointDepth = 0.0f;

        if (!projectPoint(p, screen, pointDepth)) {
            continue;
        }

        if (
            screen.x < 0 ||
            screen.x >= RENDER_WIDTH ||
            screen.y < 0 ||
            screen.y >= RENDER_HEIGHT
        ) {
            continue;
        }

        projected.push_back({
            screen,
            pointDepth
        });
    }

    // Draw point cloud.
    std::sort(
        projected.begin(),
        projected.end(),
        [](const ProjectedPoint& a,
           const ProjectedPoint& b) {
            return a.depth > b.depth;
        }
    );

    if (!projected.empty()) {

        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = std::numeric_limits<float>::lowest();

        for (const auto& p : projected) {
            minDepth = std::min(minDepth, p.depth);
            maxDepth = std::max(maxDepth, p.depth);
        }

        const float depthRange =
            std::max(0.001f, maxDepth - minDepth);

        for (const auto& p : projected) {

            const float t =
                (p.depth - minDepth) / depthRange;

            const float brightness =
                0.45f + 0.55f * (1.0f - t);

            const cv::Scalar shadedColor(
                STANDARD_POINT_COLOR[0] * brightness,
                STANDARD_POINT_COLOR[1] * brightness,
                STANDARD_POINT_COLOR[2] * brightness
            );

            cv::circle(
                visualization_,
                p.position,
                POINT_RADIUS,
                shadedColor,
                -1,
                cv::LINE_AA
            );
        }
    }

    // Camera viewing cone.
    const Point3D cameraOrigin = {
        0.0f,
        0.0f,
        0.0f
    };

    const std::vector<Point3D> frustumCorners = {
        {
            -FRUSTUM_HALF_WIDTH,
            FRUSTUM_FAR_DISTANCE,
            -FRUSTUM_HALF_HEIGHT
        },
        {
            FRUSTUM_HALF_WIDTH,
            FRUSTUM_FAR_DISTANCE,
            -FRUSTUM_HALF_HEIGHT
        },
        {
            FRUSTUM_HALF_WIDTH,
            FRUSTUM_FAR_DISTANCE,
            FRUSTUM_HALF_HEIGHT
        },
        {
            -FRUSTUM_HALF_WIDTH,
            FRUSTUM_FAR_DISTANCE,
            FRUSTUM_HALF_HEIGHT
        }
    };

    cv::Point2f cameraScreen;
    float cameraDepth = 0.0f;

    if (projectPoint(
        cameraOrigin,
        cameraScreen,
        cameraDepth
    )) {

        std::vector<cv::Point2f> corners;
        bool validCorners = true;

        for (const auto& corner : frustumCorners) {

            cv::Point2f screen;
            float depthValue = 0.0f;

            if (!projectPoint(corner, screen, depthValue)) {
                validCorners = false;
                break;
            }

            corners.push_back(screen);
        }

        if (validCorners) {

            // Draw cone edges.
            for (int i = 0; i < 4; ++i) {
                cv::line(
                    visualization_,
                    cameraScreen,
                    corners[i],
                    CAMERA_COLOR,
                    FRUSTUM_LINE_THICKNESS,
                    cv::LINE_AA
                );
            }

            // Draw far plane.
            for (int i = 0; i < 4; ++i) {
                cv::line(
                    visualization_,
                    corners[i],
                    corners[(i + 1) % 4],
                    CAMERA_COLOR,
                    FRUSTUM_LINE_THICKNESS,
                    cv::LINE_AA
                );
            }
        }
    }

    // Draw interface information.
    cv::putText(
        visualization_,
        "3D POINT CLOUD",
        cv::Point(12, 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(220, 220, 235),
        1,
        cv::LINE_AA
    );

    cv::putText(
        visualization_,
        "Yaw: " + std::to_string(CAMERA_YAW) +
        " Pitch: " + std::to_string(CAMERA_PITCH),
        cv::Point(12, 48),
        cv::FONT_HERSHEY_SIMPLEX,
        0.4,
        cv::Scalar(150, 160, 180),
        1,
        cv::LINE_AA
    );

    cv::putText(
        visualization_,
        "Points: " + std::to_string(currentPoints.size()),
        cv::Point(12, RENDER_HEIGHT - 15),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(180, 190, 210),
        1,
        cv::LINE_AA
    );
}

void PointCloudVisualizer::update(
    const cv::Mat& colorFrame,
    const cv::Mat& depthMap,
    const cv::Mat& segmentationMask
)
{
    if (depthMap.empty()) {
        currentPoints.clear();
        renderPointCloud();
        return;
    }

    generatePointCloud(
        colorFrame,
        depthMap,
        segmentationMask
    );

    renderPointCloud();
}

cv::Mat PointCloudVisualizer::getVisualization() const
{
    return visualization_.clone();
}
