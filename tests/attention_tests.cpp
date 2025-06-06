#include "mini_torch/attention.h"
#include "mini_torch/model.h"
#include <iostream>

/// @brief Run basic genesis attention forward path
int main() {
    Tensor q({2,4}, 0.5f);
    Tensor k({2,4}, 0.5f);
    Tensor v({2,4}, 1.0f);
    GenesisAttention ga(8, 2);
    auto out = ga(q, k, v);
    if(out.shape() != std::vector<size_t>{2,4}) return 1;
    std::cout << "attention tests passed\n";
    return 0;
}
