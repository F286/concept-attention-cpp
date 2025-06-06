#include "mini_torch/model.h"
#include <iostream>

/**
 * @brief Baseline model using standard attention for comparison.
 */
int main() {
    Tensor input({1, 4}, 0.5f);
    Model model(4);
    auto out = model(input);
    std::cout << "Output size: " << out.size() << "\n";
    return 0;
}
