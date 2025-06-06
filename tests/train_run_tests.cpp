#include <doctest/doctest.h>
#include "mini_torch/model.h"
#include <random>
#include <sstream>
#include <iostream>

/// @brief Execute five-epoch training and write losses to stream
static void run_training(std::ostream &os) {
    const size_t dim = 4;
    const float lr = 0.1f;
    const int epochs = 5;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Tensor input({1, dim});
    Tensor target({1, dim});
    for (size_t i = 0; i < input.size(); ++i) input[i] = dist(rng);
    for (size_t i = 0; i < target.size(); ++i) target[i] = dist(rng);

    Model baseline(dim);
    GenesisModel genesis(dim);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        baseline.train_step(input, target, lr);
        genesis.train_step(input, target, lr);
        auto out_b = baseline(input);
        auto out_g = genesis(input);
        float loss_b = 0.0f, loss_g = 0.0f;
        for (size_t i = 0; i < out_b.size(); ++i) {
            float diff_b = out_b[i] - target[i];
            float diff_g = out_g[i] - target[i];
            loss_b += 0.5f * diff_b * diff_b;
            loss_g += 0.5f * diff_g * diff_g;
        }
        os << "Epoch " << epoch << " baseline " << loss_b
           << " genesis " << loss_g << '\n';
    }
}

/// @brief Training integration test
TEST_CASE("[TRAIN] run training loop") {

    std::ostringstream buffer;
    run_training(buffer);

    std::istringstream iss(buffer.str());
    std::string line; int count = 0;
    while (std::getline(iss, line)) ++count;
    CHECK(count == 5);

    std::cout << buffer.str();
}
