#include "mini_torch/tensor.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>

Tensor::Tensor(std::vector<size_t> shape, float value, bool requires_grad)
    : m_shape(std::move(shape)), m_requires_grad(requires_grad) {
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1u, std::multiplies<>());
    size_t chunks = (m_size + floatv::size() - 1) / floatv::size();
    m_data.assign(chunks, floatv(value));
    if (m_requires_grad) {
        m_grad = std::make_unique<Tensor>(m_shape);
    }
}

Tensor::Tensor(const Tensor& other)
    : m_shape(other.m_shape), m_size(other.m_size), m_data(other.m_data),
      m_requires_grad(other.m_requires_grad) {
    if (other.m_grad)
        m_grad = std::make_unique<Tensor>(*other.m_grad);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    m_shape = other.m_shape;
    m_size = other.m_size;
    m_data = other.m_data;
    m_requires_grad = other.m_requires_grad;
    if (other.m_grad)
        m_grad = std::make_unique<Tensor>(*other.m_grad);
    else
        m_grad.reset();
    return *this;
}

ScalarRef Tensor::operator[](size_t idx) {
    auto &chunk = m_data[idx / floatv::size()];
    return ScalarRef{&chunk, idx % floatv::size()};
}

float Tensor::operator[](size_t idx) const {
    const auto &chunk = m_data[idx / floatv::size()];
    return chunk[idx % floatv::size()];
}

ScalarRef Tensor::at(size_t row, size_t col) {
    assert(m_shape.size() == 2);
    size_t cols = m_shape[1];
    return (*this)[row * cols + col];
}

float Tensor::at(size_t row, size_t col) const {
    assert(m_shape.size() == 2);
    size_t cols = m_shape[1];
    return (*this)[row * cols + col];
}

size_t Tensor::size() const {
    return m_size;
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
                sum += a[i * a.m_shape[1] + k] * b[k * b.m_shape[1] + j];
            }
            out[i * b.m_shape[1] + j] = sum;
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b) {
    assert(a.m_shape == b.m_shape);
    Tensor out(a.m_shape);
    for (size_t i = 0; i < a.m_data.size(); ++i)
        out.m_data[i] = a.m_data[i] + b.m_data[i];
    return out;
}

Tensor Tensor::sub(const Tensor &a, const Tensor &b) {
    assert(a.m_shape == b.m_shape);
    Tensor out(a.m_shape);
    for (size_t i = 0; i < a.m_data.size(); ++i)
        out.m_data[i] = a.m_data[i] - b.m_data[i];
    return out;
}

Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
    assert(a.m_shape == b.m_shape);
    Tensor out(a.m_shape);
    for (size_t i = 0; i < a.m_data.size(); ++i)
        out.m_data[i] = a.m_data[i] * b.m_data[i];
    return out;
}

Tensor Tensor::matmul(const Tensor &other) const {
    return Tensor::matmul(*this, other);
}

Tensor Tensor::operator+(const Tensor &other) const {
    return Tensor::add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
    return Tensor::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return Tensor::mul(*this, other);
}

Tensor Tensor::relu() const {
    Tensor out = *this;
    out.relu_();
    return out;
}

void Tensor::relu_() {
    const floatv zero(0.0f);
    for (auto &chunk : m_data) {
        std::experimental::where(chunk < zero, chunk) = zero;
    }
}

void Tensor::fill(float v) {
    std::fill(m_data.begin(), m_data.end(), floatv(v));
}

Tensor Tensor::transpose(const Tensor &t) {
    assert(t.m_shape.size() == 2);
    Tensor out({t.m_shape[1], t.m_shape[0]});
    for (size_t i = 0; i < t.m_shape[0]; ++i)
        for (size_t j = 0; j < t.m_shape[1]; ++j)
            out[j * t.m_shape[0] + i] = t[i * t.m_shape[1] + j];
    return out;
}

Tensor Tensor::softmax(const Tensor &t) {
    assert(t.m_shape.size() == 2);
    Tensor out(t.m_shape);
    size_t rows = t.m_shape[0];
    size_t cols = t.m_shape[1];
    for (size_t r = 0; r < rows; ++r) {
        float max_v = t[r * cols];
        for (size_t c = 1; c < cols; ++c)
            if (t[r * cols + c] > max_v)
                max_v = t[r * cols + c];
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float e = std::exp(t[r * cols + c] - max_v);
            out[r * cols + c] = e;
            sum += e;
        }
        for (size_t c = 0; c < cols; ++c)
            out[r * cols + c] /= sum;
    }
    return out;
}

Tensor &Tensor::grad() {
    if (!m_grad)
        m_grad = std::make_unique<Tensor>(m_shape);
    return *m_grad;
}

const Tensor &Tensor::grad() const {
    if (!m_grad)
        m_grad = std::make_unique<Tensor>(m_shape);
    return *m_grad;
}

bool Tensor::requires_grad() const {
    return m_requires_grad;
}

void Tensor::zero_grad() {
    if (m_grad)
        m_grad->fill(0.0f);
}
