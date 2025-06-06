#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "mini_torch/tensor.h"
#include "mini_torch/attention.h"
#include "mini_torch/model.h"
#include <random>
#include <cmath>

/// @brief Verify tensor addition
TEST_CASE("tensor add") {
    Tensor t({2, 2}, 1.0f);
    t[0] = 2.0f;
    Tensor t2({2, 2}, 3.0f);
    auto sum = Tensor::add(t, t2);
    CHECK(sum.size() == 4);
    CHECK(sum[0] == doctest::Approx(5.0f));
}

/// @brief Verify tensor matrix multiplication
TEST_CASE("tensor matmul") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    float vals_a[] = {1, 2, 3, 4, 5, 6};
    float vals_b[] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; ++i) a[i] = vals_a[i];
    for (size_t i = 0; i < 6; ++i) b[i] = vals_b[i];
    auto out = Tensor::matmul(a, b);
    CHECK(out.shape() == std::vector<size_t>{2, 2});
    CHECK(out[0] == doctest::Approx(58.0f));
    CHECK(out[1] == doctest::Approx(64.0f));
    CHECK(out[2] == doctest::Approx(139.0f));
    CHECK(out[3] == doctest::Approx(154.0f));
}

/// @brief Verify tensor relu operation
TEST_CASE("tensor relu") {
    Tensor t({1, 4});
    float vals[] = {-1.0f, 0.0f, 2.0f, -3.0f};
    for (size_t i = 0; i < 4; ++i) t[i] = vals[i];
    t.relu();
    CHECK(t[0] == doctest::Approx(0.0f));
    CHECK(t[1] == doctest::Approx(0.0f));
    CHECK(t[2] == doctest::Approx(2.0f));
    CHECK(t[3] == doctest::Approx(0.0f));
}

static_assert(TensorLike<Tensor>);

/// @brief Verify standard attention output
TEST_CASE("standard attention") {
    Tensor q({1, 2});
    Tensor k({2, 2});
    Tensor v({2, 2});
    q[0] = 1.0f; q[1] = 0.0f;
    k[0] = 1.0f; k[1] = 0.0f; k[2] = 0.0f; k[3] = 1.0f;
    v[0] = 2.0f; v[1] = 3.0f; v[2] = 4.0f; v[3] = 5.0f;
    auto out = Attention::apply(q, k, v);
    CHECK(out.shape() == std::vector<size_t>{1, 2});
    CHECK(out[0] == doctest::Approx(2.0f));
    CHECK(out[1] == doctest::Approx(3.0f));
}

/// @brief Verify genesis attention shape
TEST_CASE("genesis attention") {
    Tensor q({2, 4}, 0.5f);
    Tensor k({2, 4}, 0.5f);
    Tensor v({2, 4}, 1.0f);
    GenesisAttention ga(8, 2);
    auto out = ga(q, k, v);
    CHECK(out.shape() == std::vector<size_t>{2, 4});
}

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

