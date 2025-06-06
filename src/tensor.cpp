#include "mini_torch/tensor.h"
#include <numeric>
#include <algorithm>

Tensor::Tensor(std::vector<size_t> shape, float value)
    : m_shape(std::move(shape)), m_data(std::accumulate(m_shape.begin(), m_shape.end(), 1u, std::multiplies<>()), value) {}

float &Tensor::operator[](size_t idx) {
    return m_data[idx];
}

const float &Tensor::operator[](size_t idx) const {
    return m_data[idx];
}

size_t Tensor::size() const {
    return m_data.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return m_shape;
}

Tensor Tensor::matmul(const Tensor &a, const Tensor &b) {
    assert(a.m_shape.size() == 2 && b.m_shape.size() == 2);
    assert(a.m_shape[1] == b.m_shape[0]);
    Tensor out({a.m_shape[0], b.m_shape[1]});
    for (size_t i = 0; i < a.m_shape[0]; ++i) {
        for (size_t j = 0; j < b.m_shape[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.m_shape[1]; ++k) {
                sum += a.m_data[i * a.m_shape[1] + k] * b.m_data[k * b.m_shape[1] + j];
            }
            out.m_data[i * b.m_shape[1] + j] = sum;
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    assert(a.m_shape == b.m_shape);
    Tensor out(a.m_shape);
    for (size_t i = 0; i < a.size(); ++i) {
        out.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    return out;
}

void Tensor::relu() {
    for (auto &v : m_data) {
        if (v < 0.0f) v = 0.0f;
    }
}
