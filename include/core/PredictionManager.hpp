#pragma once

#include "core/Prediction.hpp"

#include <memory>
#include <vector>

class PredictionManager {
public:
    void addPrediction(
        std::unique_ptr<Prediction> prediction
    );

    void runPrediction(
        PredictionType type,
        const cv::Mat& frame
    );

    Prediction* getPrediction(PredictionType type);

    const Prediction* getPrediction(
        PredictionType type
    ) const;

    cv::Mat getVisualization(PredictionType type) const;

private:
    std::vector<std::unique_ptr<Prediction>> predictions_;
};