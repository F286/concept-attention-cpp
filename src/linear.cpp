#include "mini_torch/linear.h"
#include "mini_torch/optim.h"
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

void Linear::step(const Tensor &input, const Tensor &grad_output, float lr) {
    size_t batch = input.shape()[0];
    size_t in_dim = m_weight.shape()[0];
    size_t out_dim = m_weight.shape()[1];

    Tensor grad_w(m_weight.shape());
    Tensor grad_b({1, out_dim});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < in_dim; ++i) {
            for (size_t j = 0; j < out_dim; ++j) {
                size_t w_idx = i * out_dim + j;
                grad_w[w_idx] += input[b * in_dim + i] * grad_output[b * out_dim + j];
            }
        }
    }

    for (size_t j = 0; j < out_dim; ++j) {
        for (size_t b = 0; b < batch; ++b)
            grad_b[j] += grad_output[b * out_dim + j];
    }

    SGD opt(lr);
    opt.step(m_weight, grad_w);
    opt.step(m_bias, grad_b);
}

Tensor &Linear::weight() { return m_weight; }
const Tensor &Linear::weight() const { return m_weight; }
Tensor &Linear::bias() { return m_bias; }
const Tensor &Linear::bias() const { return m_bias; }
