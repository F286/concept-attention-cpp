#include "mini_torch/autograd.h"
#include "mini_torch/tensor.h"

AddFunction::AddFunction(Tensor &a, Tensor &b)
    : m_a(a), m_b(b), m_a_saved(a), m_b_saved(b) {}

void AddFunction::backward(const Tensor &grad_output) {
    if (m_a.get().requires_grad())
        m_a.get().backward(grad_output);
    if (m_b.get().requires_grad())
        m_b.get().backward(grad_output);
}

SubFunction::SubFunction(Tensor &a, Tensor &b)
    : m_a(a), m_b(b), m_a_saved(a), m_b_saved(b) {}

void SubFunction::backward(const Tensor &grad_output) {
    if (m_a.get().requires_grad())
        m_a.get().backward(grad_output);
    if (m_b.get().requires_grad()) {
        Tensor neg = grad_output;
        for (size_t i = 0; i < neg.size(); ++i)
            neg[i] = -neg[i];
        m_b.get().backward(neg);
    }
}

MulFunction::MulFunction(Tensor &a, Tensor &b)
    : m_a(a), m_b(b), m_a_saved(a), m_b_saved(b) {}

void MulFunction::backward(const Tensor &grad_output) {
    if (m_a.get().requires_grad()) {
        Tensor grad_a = Tensor::mul(grad_output, m_b_saved);
        m_a.get().backward(grad_a);
    }
    if (m_b.get().requires_grad()) {
        Tensor grad_b = Tensor::mul(grad_output, m_a_saved);
        m_b.get().backward(grad_b);
    }
}

MatMulFunction::MatMulFunction(Tensor &a, Tensor &b)
    : m_a(a), m_b(b), m_a_saved(a), m_b_saved(b) {}

void MatMulFunction::backward(const Tensor &grad_output) {
    if (m_a.get().requires_grad()) {
        Tensor grad_a = Tensor::matmul(grad_output, Tensor::transpose(m_b_saved));
        m_a.get().backward(grad_a);
    }
    if (m_b.get().requires_grad()) {
        Tensor grad_b = Tensor::matmul(Tensor::transpose(m_a_saved), grad_output);
        m_b.get().backward(grad_b);
    }
}

ReLUFunction::ReLUFunction(Tensor &a, Tensor mask) : m_a(a), m_mask(std::move(mask)) {}

void ReLUFunction::backward(const Tensor &grad_output) {
    if (!m_a.get().requires_grad()) return;
    Tensor grad = grad_output * m_mask;
    m_a.get().backward(grad);
}

TransposeFunction::TransposeFunction(Tensor &a) : m_a(a) {}

void TransposeFunction::backward(const Tensor &grad_output) {
    if (!m_a.get().requires_grad()) return;
    Tensor grad = Tensor::transpose(grad_output);
    m_a.get().backward(grad);
}

SoftmaxFunction::SoftmaxFunction(Tensor &a, Tensor output)
    : m_a(a), m_output(std::move(output)) {}

void SoftmaxFunction::backward(const Tensor &grad_output) {
    if (!m_a.get().requires_grad()) return;
    Tensor grad(m_a.get().shape());
    size_t rows = m_a.get().shape()[0];
    size_t cols = m_a.get().shape()[1];
    for (size_t r = 0; r < rows; ++r) {
        float dot = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            dot += grad_output.at(r, c) * m_output.at(r, c);
        for (size_t c = 0; c < cols; ++c)
            grad.at(r, c) = m_output.at(r, c) * (grad_output.at(r, c) - dot);
    }
    m_a.get().backward(grad);
}
