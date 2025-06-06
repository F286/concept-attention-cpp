#include "mini_torch/linear.h"
#include "mini_torch/optim.h"
#include <random>

Linear::Linear(size_t in_features, size_t out_features)
    : m_weight({in_features, out_features}, 0.0f, true),
      m_bias({1, out_features}, 0.0f, true) {
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


Tensor &Linear::weight() { return m_weight; }
const Tensor &Linear::weight() const { return m_weight; }
Tensor &Linear::bias() { return m_bias; }
const Tensor &Linear::bias() const { return m_bias; }

std::vector<Tensor*> Linear::parameters() {
    return {&m_weight, &m_bias};
}
