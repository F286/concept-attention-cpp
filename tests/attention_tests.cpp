#include "mini_torch/attention.h"
#include <iostream>

/// @brief Verify standard attention
static int test_standard() {
    Tensor q({1, 2});
    Tensor k({2, 2});
    Tensor v({2, 2});
    q[0] = 1.0f; q[1] = 0.0f;
    k[0] = 1.0f; k[1] = 0.0f; k[2] = 0.0f; k[3] = 1.0f;
    v[0] = 2.0f; v[1] = 3.0f; v[2] = 4.0f; v[3] = 5.0f;
    auto out = Attention::apply(q, k, v);
    if (out.shape() != std::vector<size_t>{1, 2}) return 1;
    if (out[0] != 2.0f || out[1] != 3.0f) return 2;
    return 0;
}

/// @brief Verify genesis attention shape
static int test_genesis() {
    Tensor q({2, 4}, 0.5f);
    Tensor k({2, 4}, 0.5f);
    Tensor v({2, 4}, 1.0f);
    GenesisAttention ga(8, 2);
    auto out = ga(q, k, v);
    if (out.shape() != std::vector<size_t>{2, 4}) return 1;
    return 0;
}

/// @brief Entry point for attention tests
int main() {
    if (int r = test_standard()) return r;
    if (int r = test_genesis()) return r + 10;
    std::cout << "attention tests passed\n";
    return 0;
}
