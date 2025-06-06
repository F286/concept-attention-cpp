#include "mini_torch/model.h"
#include <iostream>

/// @brief Run baseline model demonstration
int run_baseline_demo() {
    Tensor input({1, 4}, 0.5f);
    Model model(4);
    auto out = model(input);
    std::cout << "Output size: " << out.size() << "\n";
    return 0;
}

