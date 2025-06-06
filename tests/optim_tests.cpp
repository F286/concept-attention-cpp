#include <doctest/doctest.h>
#include "mini_torch/optim.h"
#include <type_traits>

/// @brief verify SGD parameter update
TEST_CASE("sgd step") {
    Tensor param({1,2});
    param[0] = 1.0f; param[1] = -1.0f;
    Tensor grad({1,2});
    grad[0] = 0.5f; grad[1] = -0.5f;
    SGD opt(0.1f);
    opt.step(param, grad);
    CHECK(param[0] == doctest::Approx(0.95f));
    CHECK(param[1] == doctest::Approx(-0.95f));
}

static_assert(std::is_copy_constructible_v<SGD>);
