#include <doctest/doctest.h>
#include "mini_torch/model.h"
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
        loss_b=0.0f; loss_g=0.0f;
        for(size_t i=0;i<out_b.size();++i){
            loss_b += 0.5f*(out_b[i]-target[i])*(out_b[i]-target[i]);
            loss_g += 0.5f*(out_g[i]-target[i])*(out_g[i]-target[i]);
        }
    }
    CHECK(std::isfinite(loss_b));
    CHECK(std::isfinite(loss_g));
}
