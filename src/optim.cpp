#include "mini_torch/optim.h"
#include <cassert>

SGD::SGD(std::vector<Tensor*> params, float lr) : m_params(std::move(params)), m_lr(lr) {}

void SGD::step() {
    for (Tensor* p : m_params) {
        Tensor &grad = p->grad();
        assert(p->shape() == grad.shape());
        for (size_t i = 0; i < p->size(); ++i)
            (*p)[i] -= m_lr * grad[i];
    }
}

void SGD::zero_grad() {
    for (Tensor* p : m_params)
        p->zero_grad();
}
