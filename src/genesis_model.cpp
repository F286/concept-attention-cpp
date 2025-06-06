#include "mini_torch/model.h"
#include <iostream>

/**
 * @brief Implementation of a model using genesis attention.
 * It mirrors the baseline model but substitutes the attention
 * mechanism with the experimental version.
 */
int main() {
    Tensor input({1, 4}, 0.5f);
    GenesisModel model(4);
    auto out = model(input);
    std::cout << "Output size: " << out.size() << "\n";
    return 0;
}
