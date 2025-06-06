#include <doctest/doctest.h>
#include "mini_torch/dropout.h"
#include "mini_torch/tensor.h"
#include <concepts>
#include <random>

/// @brief dropout returns input in eval mode
TEST_CASE("dropout eval mode") {
    Dropout d(0.5f);
    d.train(false);
    Tensor x({1,2});
    x[0] = 1.0f; x[1] = 2.0f;
    Tensor out = d(x);
    CHECK(out[0] == doctest::Approx(1.0f));
    CHECK(out[1] == doctest::Approx(2.0f));
}

/// @brief p=0 leaves tensor unchanged
TEST_CASE("dropout no drop") {
    Dropout d(0.0f);
    Tensor x({1,2});
    x[0] = 1.0f; x[1] = 2.0f;
    Tensor out = d(x);
    CHECK(out[0] == doctest::Approx(1.0f));
    CHECK(out[1] == doctest::Approx(2.0f));
}

/// @brief p=1 zeros out tensor
TEST_CASE("dropout all drop") {
    Dropout d(1.0f);
    Tensor x({1,3});
    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
    Tensor out = d(x);
    CHECK(out[0] == doctest::Approx(0.0f));
    CHECK(out[1] == doctest::Approx(0.0f));
    CHECK(out[2] == doctest::Approx(0.0f));
}

/// @brief verify deterministic behaviour for 0.5 probability
TEST_CASE("dropout deterministic half") {
    Dropout d(0.5f);
    Tensor x({1,4});
    for(size_t i=0;i<4;++i) x[i] = static_cast<float>(i+1);
    // replicate distribution
    std::mt19937 rng(42);
    std::bernoulli_distribution bern(0.5);
    Tensor expected({1,4});
    float scale = 1.0f / 0.5f;
    for(size_t i=0;i<4;++i) {
        bool keep = bern(rng);
        expected[i] = keep ? x[i]*scale : 0.0f;
    }
    Tensor out = d(x);
    for(size_t i=0;i<4;++i)
        CHECK(out[i] == doctest::Approx(expected[i]));
}

static_assert(std::derived_from<Dropout, Module>);
