#include <doctest/doctest.h>
#include "mini_torch/loss.h"
#include <type_traits>

/// @brief verify mse loss forward and backward
TEST_CASE("mse loss") {
    Tensor pred({1,2});
    pred[0] = 1.0f; pred[1] = 2.0f;
    Tensor target({1,2});
    target[0] = 0.0f; target[1] = 0.0f;
    MSELoss loss;
    float v = loss(pred, target);
    CHECK(v == doctest::Approx(2.5f));
    pred = Tensor({1,2},0.0f,true);
    pred[0]=1.0f; pred[1]=2.0f;
    v = loss(pred, target);
    loss.backward(pred, target);
    CHECK(pred.grad()[0] == doctest::Approx(1.0f));
    CHECK(pred.grad()[1] == doctest::Approx(2.0f));
}

static_assert(std::is_copy_constructible_v<MSELoss>);

/// @brief verify cross entropy loss forward and backward
TEST_CASE("cross entropy loss") {
    Tensor logits({2,3});
    logits[0] = 1.0f; logits[1] = 2.0f; logits[2] = 3.0f;
    logits[3] = 1.0f; logits[4] = 2.0f; logits[5] = 3.0f;
    std::vector<size_t> target = {2, 0};
    CrossEntropyLoss loss;
    float v = loss(logits, target);
    CHECK(v > 0.0f);
    logits = Tensor({2,3},0.0f,true);
    logits[0]=1.0f; logits[1]=2.0f; logits[2]=3.0f;
    logits[3]=1.0f; logits[4]=2.0f; logits[5]=3.0f;
    v = loss(logits, target);
    loss.backward(logits, target);
    auto &grad = logits.grad();
    CHECK(grad.shape() == logits.shape());
    CHECK(grad.at(0,0) + grad.at(0,1) + grad.at(0,2) == doctest::Approx(0.0f));
    CHECK(grad.at(1,0) + grad.at(1,1) + grad.at(1,2) == doctest::Approx(0.0f));
}

static_assert(std::is_copy_constructible_v<CrossEntropyLoss>);
