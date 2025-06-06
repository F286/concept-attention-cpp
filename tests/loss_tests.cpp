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
    auto grad = loss.backward(pred, target);
    CHECK(grad.shape() == pred.shape());
    CHECK(grad[0] == doctest::Approx(1.0f));
    CHECK(grad[1] == doctest::Approx(2.0f));
}

static_assert(std::is_copy_constructible_v<MSELoss>);
