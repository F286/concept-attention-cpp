#include "mini_torch/loss.h"
#include <cassert>
#include <cmath>

float MSELoss::operator()(const Tensor &pred, const Tensor &target) const {
    assert(pred.shape() == target.shape());
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / pred.size();
}

void MSELoss::backward(const Tensor &pred, const Tensor &target) const {
    assert(pred.shape() == target.shape());
    Tensor &grad = const_cast<Tensor&>(pred).grad();
    grad.zero_grad();
    for (size_t i = 0; i < pred.size(); ++i)
        grad[i] += 2.0f * (pred[i] - target[i]) / static_cast<float>(pred.size());
}

float CrossEntropyLoss::operator()(const Tensor &logits, const std::vector<size_t> &target) const {
    assert(logits.shape().size() == 2);
    size_t n = logits.shape()[0];
    size_t c = logits.shape()[1];
    assert(target.size() == n);

    float loss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float max_logit = logits.at(i,0);
        for (size_t j = 1; j < c; ++j)
            if (logits.at(i,j) > max_logit) max_logit = logits.at(i,j);
        float sum = 0.0f;
        for (size_t j = 0; j < c; ++j)
            sum += std::exp(logits.at(i,j) - max_logit);
        float log_prob = logits.at(i, target[i]) - max_logit - std::log(sum);
        loss -= log_prob;
    }
    return loss / static_cast<float>(n);
}

void CrossEntropyLoss::backward(const Tensor &logits, const std::vector<size_t> &target) const {
    assert(logits.shape().size() == 2);
    size_t n = logits.shape()[0];
    size_t c = logits.shape()[1];
    assert(target.size() == n);
    Tensor &grad = const_cast<Tensor&>(logits).grad();
    grad.zero_grad();
    for (size_t i = 0; i < n; ++i) {
        float max_logit = logits.at(i,0);
        for (size_t j = 1; j < c; ++j)
            if (logits.at(i,j) > max_logit) max_logit = logits.at(i,j);
        float sum = 0.0f;
        for (size_t j = 0; j < c; ++j)
            sum += std::exp(logits.at(i,j) - max_logit);
        for (size_t j = 0; j < c; ++j) {
            float soft = std::exp(logits.at(i,j) - max_logit) / sum;
            grad.at(i,j) += (soft - (j == target[i] ? 1.0f : 0.0f)) / static_cast<float>(n);
        }
    }
}
