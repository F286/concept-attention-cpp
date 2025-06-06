#include "mini_torch/model.h"

Model::Model(size_t dim)
    : m_proj_q(dim, dim), m_proj_k(dim, dim), m_proj_v(dim, dim), m_out(dim, dim) {}

Tensor Model::operator()(const Tensor &input) const {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = Attention::apply(q, k, v);
    return m_out(attn);
}

void Model::train_step(const Tensor &input, const Tensor &target, float lr) {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = Attention::apply(q, k, v);
    auto out = m_out(attn);
    Tensor grad(out.shape());
    for (size_t i = 0; i < out.size(); ++i) grad[i] = out[i] - target[i];
    m_out.step(attn, grad, lr);
}

GenesisModel::GenesisModel(size_t dim)
    : m_proj_q(dim, dim), m_proj_k(dim, dim), m_proj_v(dim, dim), m_out(dim, dim), m_attn(8, dim) {}

Tensor GenesisModel::operator()(const Tensor &input) const {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = m_attn(q, k, v);
    return m_out(attn);
}

void GenesisModel::train_step(const Tensor &input, const Tensor &target, float lr) {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = m_attn(q, k, v);
    auto out = m_out(attn);
    Tensor grad(out.shape());
    for (size_t i = 0; i < out.size(); ++i) grad[i] = out[i] - target[i];
    m_out.step(attn, grad, lr);
}
