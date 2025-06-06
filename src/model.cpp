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

GenesisModel::GenesisModel(size_t dim)
    : m_proj_q(dim, dim), m_proj_k(dim, dim), m_proj_v(dim, dim), m_out(dim, dim), m_attn(8, 16) {}

Tensor GenesisModel::operator()(const Tensor &input) const {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = m_attn(q, k, v);
    return m_out(attn);
}
