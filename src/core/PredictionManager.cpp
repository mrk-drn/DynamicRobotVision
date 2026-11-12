#include "core/PredictionManager.hpp"

#include <stdexcept>

void PredictionManager::addPrediction(
    std::unique_ptr<Prediction> prediction
) {
    if (!prediction) {
        throw std::invalid_argument("Null prediction.");
    }

    for (const auto& existing : predictions_) {
        if (existing->type() == prediction->type()) {
            throw std::runtime_error(
                "Prediction type already registered."
            );
        }
    }

    predictions_.push_back(std::move(prediction));
}

void PredictionManager::runPrediction(
    PredictionType type,
    const cv::Mat& frame
) {
    Prediction* prediction = getPrediction(type);

    if (!prediction) {
        throw std::runtime_error(
            "Prediction not registered."
        );
    }

    prediction->predict(frame);
}

Prediction* PredictionManager::getPrediction(
    PredictionType type
) {
    for (auto& prediction : predictions_) {
        if (prediction->type() == type) {
            return prediction.get();
        }
    }

    return nullptr;
}

const Prediction* PredictionManager::getPrediction(
    PredictionType type
) const {
    for (const auto& prediction : predictions_) {
        if (prediction->type() == type) {
            return prediction.get();
        }
    }

    return nullptr;
}

cv::Mat PredictionManager::getVisualization(
    PredictionType type
) const {
    const Prediction* prediction = getPrediction(type);

    if (!prediction || !prediction->hasResult()) {
        return {};
    }

    return prediction->visualize();
}