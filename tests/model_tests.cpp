#include <doctest/doctest.h>
#include "mini_torch/model.h"
#include "mini_torch/loss.h"
#include <random>
#include <cmath>

/// @brief Baseline model forward pass
TEST_CASE("baseline model demo") {
    Tensor input({4, 4}, 0.5f);
    Model model(4);
    auto out = model(input);
    CHECK(out.size() == 16);
}

/// @brief Genesis model forward pass
TEST_CASE("genesis model demo") {
    Tensor input({4, 4}, 0.5f);
    GenesisModel model(4);
    auto out = model(input);
    CHECK(out.size() == 16);
}

/// @brief Small training loop executes
TEST_CASE("training loop") {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Tensor input({4,4});
    Tensor target({4,4});
    for (size_t i=0;i<input.size();++i) input[i]=dist(rng);
    for (size_t i=0;i<target.size();++i) target[i]=dist(rng);

    Model baseline(4);
    GenesisModel genesis(4);
    const float lr = 0.1f;
    float loss_b=0.0f, loss_g=0.0f;
    for(int epoch=0; epoch<2; ++epoch){
        baseline.train_step(input, target, lr);
        genesis.train_step(input, target, lr);
        auto out_b = baseline(input);
        auto out_g = genesis(input);
        MSELoss loss;
        loss_b = loss(out_b, target);
        loss_g = loss(out_g, target);
    }
    CHECK(std::isfinite(loss_b));
    CHECK(std::isfinite(loss_g));
}

/// @brief Single train step lowers loss
TEST_CASE("model step lowers loss") {
    Tensor input({1, 2});
    input[0] = 0.5f;
    input[1] = -0.5f;
    Tensor target({1, 2});
    target.fill(0.0f);

    Model model(2);
    auto out0 = model(input);
    MSELoss loss;
    float loss0 = loss(out0, target);

    model.train_step(input, target, 0.1f);
    auto out1 = model(input);
    float loss1 = loss(out1, target);

    CHECK(loss1 < loss0);
}
