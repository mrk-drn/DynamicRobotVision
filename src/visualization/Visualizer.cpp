#include "visualization/Visualizer.hpp"

cv::Mat Visualizer::compose(
    const cv::Mat& frame,
    const std::vector<cv::Mat>& predictions,
    const std::vector<std::string>& labels,
    double scale
) {
    if (frame.empty()) {
        return {};
    }

    cv::Mat original;

    cv::resize(
        frame,
        original,
        cv::Size(),
        scale,
        scale
    );

    cv::putText(
        original,
        "Original Frame",
        cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );

    std::vector<cv::Mat> panels = {original};

    for (size_t i = 0; i < predictions.size(); ++i) {

        if (predictions[i].empty()) {
            continue;
        }

        cv::Mat panel = predictions[i].clone();

        if (panel.channels() == 1) {
            cv::cvtColor(
                panel,
                panel,
                cv::COLOR_GRAY2BGR
            );
        }
        else if (panel.channels() == 4) {
            cv::cvtColor(
                panel,
                panel,
                cv::COLOR_BGRA2BGR
            );
        }

        const double resizeScale =
            static_cast<double>(original.rows) / panel.rows;

        cv::resize(
            panel,
            panel,
            cv::Size(),
            resizeScale,
            resizeScale
        );

        const std::string label =
            i < labels.size() ? labels[i] : "Prediction";

        cv::putText(
            panel,
            label,
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            2
        );

        panels.push_back(panel);
    }

    cv::Mat combined;

    cv::hconcat(panels, combined);

    return combined;
}