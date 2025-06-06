#include "mini_torch/tensor.h"
#include <concepts>
#include <iostream>

/// @brief Verify tensor addition
static int test_add() {
    Tensor t({2, 2}, 1.0f);
    t[0] = 2.0f;
    Tensor t2({2, 2}, 3.0f);
    auto sum = Tensor::add(t, t2);
    if (sum.size() != 4) return 1;
    if (sum[0] != 5.0f) return 2;
    return 0;
}

/// @brief Verify tensor matrix multiplication
static int test_matmul() {
    Tensor a({2, 3});
    Tensor b({3, 2});
    float vals_a[] = {1, 2, 3, 4, 5, 6};
    float vals_b[] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; ++i) a[i] = vals_a[i];
    for (size_t i = 0; i < 6; ++i) b[i] = vals_b[i];
    auto out = Tensor::matmul(a, b);
    if (out.shape() != std::vector<size_t>{2, 2}) return 1;
    if (out[0] != 58.0f || out[1] != 64.0f || out[2] != 139.0f || out[3] != 154.0f)
        return 2;
    return 0;
}

/// @brief Verify tensor relu operation
static int test_relu() {
    Tensor t({1, 4});
    float vals[] = {-1.0f, 0.0f, 2.0f, -3.0f};
    for (size_t i = 0; i < 4; ++i) t[i] = vals[i];
    t.relu();
    if (t[0] != 0.0f || t[1] != 0.0f || t[2] != 2.0f || t[3] != 0.0f) return 1;
    return 0;
}

/// @brief Verify tensor operations and concept conformance
int main() {
    static_assert(TensorLike<Tensor>);
    if (int r = test_add()) return r;
    if (int r = test_matmul()) return r + 10;
    if (int r = test_relu()) return r + 20;
    std::cout << "tensor tests passed\n";
    return 0;
}
