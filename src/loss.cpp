#include "mini_torch/loss.h"
#include <cassert>

float MSELoss::operator()(const Tensor &pred, const Tensor &target) const {
    assert(pred.shape() == target.shape());
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / pred.size();
}

Tensor MSELoss::backward(const Tensor &pred, const Tensor &target) const {
    assert(pred.shape() == target.shape());
    Tensor grad(pred.shape());
    for (size_t i = 0; i < pred.size(); ++i)
        grad[i] = 2.0f * (pred[i] - target[i]) / static_cast<float>(pred.size());
    return grad;
}
