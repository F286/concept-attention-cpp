#include "mini_torch/dropout.h"
#include <random>

Dropout::Dropout(float p)
    : m_p(p), m_training(true), m_rng(42) {}

Tensor Dropout::operator()(const Tensor &input) const {
    if(!m_training || m_p == 0.0f)
        return input;
    Tensor out(input.shape());
    std::bernoulli_distribution bern(1.0f - m_p);
    float scale = 1.0f / (1.0f - m_p);
    for(size_t i=0;i<input.size();++i){
        bool keep = bern(m_rng);
        out[i] = keep ? input[i] * scale : 0.0f;
    }
    return out;
}

void Dropout::train(bool mode){ m_training = mode; }

std::vector<Tensor*> Dropout::parameters(){
    return {};
}
