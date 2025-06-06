#include <doctest/doctest.h>
#include "mini_torch/tensor.h"
#include "mini_torch/autograd.h"

/// @brief verify basic autograd on addition
TEST_CASE("autograd add") {
    Tensor x({1,1}, 1.0f, true);
    Tensor y = x + x;
    y.backward();
    CHECK(x.grad()[0] == doctest::Approx(2.0f));
}

/// @brief verify autograd on matrix multiplication
TEST_CASE("autograd matmul") {
    Tensor a({1,2}, 0.0f, true);
    a[0]=1.0f; a[1]=2.0f;
    Tensor w({2,1}, 0.0f, true);
    w[0]=3.0f; w[1]=4.0f;
    Tensor out = a.matmul(w);
    out.backward();
    CHECK(w.grad()[0] == doctest::Approx(1.0f));
    CHECK(w.grad()[1] == doctest::Approx(2.0f));
    CHECK(a.grad()[0] == doctest::Approx(3.0f));
    CHECK(a.grad()[1] == doctest::Approx(4.0f));
}

/// @brief verify autograd on transpose
TEST_CASE("autograd transpose") {
    Tensor x({2,3}, 0.0f, true);
    Tensor y = Tensor::transpose(x);
    y.backward();
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(x.grad()[i] == doctest::Approx(1.0f));
}

/// @brief verify autograd on softmax
TEST_CASE("autograd softmax") {
    Tensor x({1,2}, 0.0f, true);
    x[0] = 1.0f; x[1] = 2.0f;
    Tensor y = Tensor::softmax(x);
    Tensor g({1,2});
    g[0] = 1.0f; g[1] = 0.0f;
    y.backward(g);
    float y0 = y[0];
    float y1 = y[1];
    CHECK(x.grad()[0] == doctest::Approx(y0 * (1.0f - y0)));
    CHECK(x.grad()[1] == doctest::Approx(-y0 * y1));
}

static_assert(std::movable<AddFunction>);
static_assert(std::movable<TransposeFunction>);
static_assert(std::movable<SoftmaxFunction>);
