#include <doctest/doctest.h>
#include "mini_torch/tensor.h"

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
