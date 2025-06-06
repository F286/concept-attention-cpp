#include "mini_torch/optim.h"
#include <cassert>

SGD::SGD(float lr) : m_lr(lr) {}

void SGD::step(Tensor &param, const Tensor &grad) const {
    assert(param.shape() == grad.shape());
    for (size_t i = 0; i < param.size(); ++i)
        param[i] -= m_lr * grad[i];
}
