#pragma once

namespace PointCloudConfig
{
    // Render resolution
    inline constexpr int RENDER_WIDTH = 640;
    inline constexpr int RENDER_HEIGHT = 360;

    // Virtual camera
    inline constexpr float CAMERA_YAW = 0.0f;
    inline constexpr float CAMERA_PITCH = 0.10f;
    inline constexpr float CAMERA_FOCAL_LENGTH = 650.0f;
    inline constexpr float CAMERA_DISTANCE = 1500.0f;

    // Projection
    inline constexpr float PROJECTION_SCALE = 0.30f;

    // Frustum
    inline constexpr float FRUSTUM_FAR_DISTANCE = 2000.0f;
    inline constexpr float FRUSTUM_HALF_WIDTH = 750.0f;
    inline constexpr float FRUSTUM_HALF_HEIGHT = 450.0f;
    inline constexpr int FRUSTUM_LINE_THICKNESS = 2;

    // Point rendering
    inline constexpr int POINT_RADIUS = 1;

    inline constexpr float STANDARD_BLUE = 80.0f;
    inline constexpr float STANDARD_GREEN = 200.0f;
    inline constexpr float STANDARD_RED = 255.0f;

    // Background (BGR)
    inline constexpr unsigned char BACKGROUND_BLUE = 25;
    inline constexpr unsigned char BACKGROUND_GREEN = 15;
    inline constexpr unsigned char BACKGROUND_RED = 15;
}
