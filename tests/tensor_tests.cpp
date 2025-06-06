#include <doctest/doctest.h>
#include "mini_torch/tensor.h"

/// @brief Verify tensor addition
TEST_CASE("tensor add") {
    Tensor t({2, 2}, 1.0f);
    t[0] = 2.0f;
    Tensor t2({2, 2}, 3.0f);
    auto sum = t + t2;
    CHECK(sum.size() == 4);
    CHECK(sum[0] == doctest::Approx(5.0f));
}

/// @brief Verify tensor subtract and multiply
TEST_CASE("tensor sub mul") {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(1.5f);
    auto diff = a - b;
    auto prod = a * b;
    CHECK(diff[0] == doctest::Approx(0.5f));
    CHECK(prod[0] == doctest::Approx(3.0f));
}

/// @brief Verify tensor matrix multiplication
TEST_CASE("tensor matmul") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    float vals_a[] = {1, 2, 3, 4, 5, 6};
    float vals_b[] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; ++i) a[i] = vals_a[i];
    for (size_t i = 0; i < 6; ++i) b[i] = vals_b[i];
    auto out = a.matmul(b);
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
    t.relu_();
    CHECK(t[0] == doctest::Approx(0.0f));
    CHECK(t[1] == doctest::Approx(0.0f));
    CHECK(t[2] == doctest::Approx(2.0f));
    CHECK(t[3] == doctest::Approx(0.0f));
}

/// @brief Verify 2D indexing and fill
TEST_CASE("tensor at and fill") {
    Tensor t({2, 3});
    t.fill(0.0f);
    t.at(1, 2) = 5.0f;
    CHECK(t[5] == doctest::Approx(5.0f));
    CHECK(t.at(1, 2) == doctest::Approx(5.0f));
}

/// @brief Verify transpose and softmax
TEST_CASE("tensor transpose softmax") {
    Tensor t({2, 2});
    t[0] = 1.0f; t[1] = 2.0f;
    t[2] = 3.0f; t[3] = 4.0f;
    auto tt = Tensor::transpose(t);
    CHECK(tt.shape() == std::vector<size_t>{2, 2});
    CHECK(tt[0] == doctest::Approx(1.0f));
    CHECK(tt[1] == doctest::Approx(3.0f));
    CHECK(tt[2] == doctest::Approx(2.0f));
    CHECK(tt[3] == doctest::Approx(4.0f));

    auto soft = Tensor::softmax(t);
    CHECK(soft.shape() == t.shape());
    CHECK(soft[0] + soft[1] == doctest::Approx(1.0f));
    CHECK(soft[2] + soft[3] == doctest::Approx(1.0f));
}

static_assert(TensorLike<Tensor>);
