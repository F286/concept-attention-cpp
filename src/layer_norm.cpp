#include "mini_torch/layer_norm.h"
#include <cassert>
#include <cmath>

LayerNorm::LayerNorm(size_t features, float eps)
    : m_features(features), m_eps(eps),
      m_gamma({1, features}, 1.0f, true),
      m_beta({1, features}, 0.0f, true) {}

Tensor LayerNorm::operator()(const Tensor &input) const {
    assert(input.shape().size() == 2);
    assert(input.shape()[1] == m_features);
    Tensor out(input.shape());
    size_t batch = input.shape()[0];
    size_t dim = m_features;
    for(size_t b=0;b<batch;++b){
        float mean = 0.0f;
        for(size_t f=0;f<dim;++f)
            mean += input.at(b,f);
        mean /= static_cast<float>(dim);
        float var = 0.0f;
        for(size_t f=0;f<dim;++f){
            float diff = input.at(b,f) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(dim);
        float denom = 1.0f / std::sqrt(var + m_eps);
        for(size_t f=0;f<dim;++f){
            float norm = (input.at(b,f) - mean) * denom;
            out.at(b,f) = norm * m_gamma[f] + m_beta[f];
        }
    }
    return out;
}

std::vector<Tensor*> LayerNorm::parameters(){
    return {&m_gamma, &m_beta};
}

Tensor &LayerNorm::weight(){ return m_gamma; }
const Tensor &LayerNorm::weight() const{ return m_gamma; }
Tensor &LayerNorm::bias(){ return m_beta; }
const Tensor &LayerNorm::bias() const{ return m_beta; }
