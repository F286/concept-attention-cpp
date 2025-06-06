#include "mini_torch/model.h"
#include <iostream>

/**
 * @brief Run model using genesis attention for comparison
 */
int run_genesis_demo() {
    Tensor input({1, 4}, 0.5f);
    GenesisModel model(4);
    auto out = model(input);
    std::cout << "Output size: " << out.size() << "\n";
    return 0;
}

