#include "mini_torch/linear.h"
#include <random>

Linear::Linear(size_t in, size_t out)
    : m_weight({in, out}), m_bias({1, out}) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < m_weight.size(); ++i) m_weight[i] = dist(rng);
    for (size_t i = 0; i < m_bias.size(); ++i) m_bias[i] = dist(rng);
}

Tensor Linear::operator()(const Tensor &input) const {
    auto out = Tensor::matmul(input, m_weight);
    for (size_t i = 0; i < out.size(); ++i) out[i] += m_bias[i % m_bias.shape()[1]];
    return out;
}
