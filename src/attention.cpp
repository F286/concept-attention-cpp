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
    : m_concepts(concepts), m_dim(dim) {}

Tensor GenesisAttention::operator()(const Tensor &q, const Tensor &k, const Tensor &v) const {
    std::vector<Tensor> sims;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t c = 0; c < m_concepts; ++c) {
        Tensor wq({q.shape()[1], m_dim});
        Tensor wk({k.shape()[1], m_dim});
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = dist(rng);
        for (size_t i = 0; i < wk.size(); ++i) wk[i] = dist(rng);
        auto q_p = Tensor::matmul(q, wq);
        auto k_p = Tensor::matmul(k, wk);
        sims.push_back(Tensor::matmul(q_p, k_p));
    }
    Tensor min_sim = sims[0];
    for (size_t i = 1; i < sims.size(); ++i) {
        for (size_t j = 0; j < min_sim.size(); ++j)
            if (sims[i][j] < min_sim[j]) min_sim[j] = sims[i][j];
    }
    return Tensor::matmul(min_sim, v);
}
