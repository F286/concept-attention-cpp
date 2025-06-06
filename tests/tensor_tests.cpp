#include "mini_torch/tensor.h"
#include <iostream>
#include <concepts>

/// @brief Verify tensor operations and concept conformance
int main() {
    static_assert(TensorLike<Tensor>);
    Tensor t({2,2}, 1.0f);
    t[0] = 2.0f;
    Tensor t2({2,2}, 3.0f);
    auto sum = Tensor::add(t, t2);
    if(sum.size() != 4) return 1;
    if(sum[0] != 5.0f) return 2;
    std::cout << "tensor tests passed\n";
    return 0;
}
