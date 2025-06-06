#include "mini_torch/attention.h"
#include <cmath>
#include <random>

Tensor Attention::apply(const Tensor &q, const Tensor &k, const Tensor &v) {
    auto kt = Tensor::transpose(k);
    auto scores = Tensor::matmul(q, kt);
    float scale = 1.0f / std::sqrt(static_cast<float>(q.shape()[1]));
    for (size_t i = 0; i < scores.size(); ++i) scores[i] *= scale;
    auto probs = Tensor::softmax(scores);
    return Tensor::matmul(probs, v);
}

GenesisAttention::GenesisAttention(size_t concepts, size_t dim)
    : m_initialized(false), m_concepts(concepts), m_dim(dim),
      m_wq(concepts), m_wk(concepts), m_rng(42) {}

Tensor GenesisAttention::operator()(const Tensor &q, const Tensor &k, const Tensor &v) const {
    if (!m_initialized) {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (size_t c = 0; c < m_concepts; ++c) {
            m_wq[c] = Tensor({q.shape()[1], m_dim});
            m_wk[c] = Tensor({k.shape()[1], m_dim});
            for (size_t i = 0; i < m_wq[c].size(); ++i) m_wq[c][i] = dist(m_rng);
            for (size_t i = 0; i < m_wk[c].size(); ++i) m_wk[c][i] = dist(m_rng);
        }
        m_initialized = true;
    }

    auto q_p = Tensor::matmul(q, m_wq[0]);
    auto k_p = Tensor::matmul(k, m_wk[0]);
    auto min_sim = Tensor::matmul(q_p, Tensor::transpose(k_p));
    for (size_t c = 1; c < m_concepts; ++c) {
        auto q_pc = Tensor::matmul(q, m_wq[c]);
        auto k_pc = Tensor::matmul(k, m_wk[c]);
        auto sim = Tensor::matmul(q_pc, Tensor::transpose(k_pc));
        for (size_t i = 0; i < sim.size(); ++i)
            if (sim[i] < min_sim[i]) min_sim[i] = sim[i];
    }

    return Tensor::matmul(min_sim, v);
}

std::vector<Tensor*> GenesisAttention::parameters() {
    if (!m_initialized) return {};
    std::vector<Tensor*> params;
    for (auto &t : m_wq) params.push_back(&t);
    for (auto &t : m_wk) params.push_back(&t);
    return params;
}
