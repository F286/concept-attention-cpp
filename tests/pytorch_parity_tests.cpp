#include <doctest/doctest.h>
#include "mini_torch/tensor.h"
#include "mini_torch/loss.h"

/// @brief Validate tensor operations against PyTorch values
TEST_CASE("tensor parity with pytorch") {
    Tensor m({2,2});
    m[0] = 0.1f; m[1] = 0.2f;
    m[2] = 0.3f; m[3] = -0.1f;

    auto soft = Tensor::softmax(m);
    CHECK(soft[0] == doctest::Approx(0.4750208f));
    CHECK(soft[1] == doctest::Approx(0.5249792f));
    CHECK(soft[2] == doctest::Approx(0.5986876f));
    CHECK(soft[3] == doctest::Approx(0.4013123f));

    Tensor r = m;
    r = r.relu();
    CHECK(r[0] == doctest::Approx(0.1f));
    CHECK(r[1] == doctest::Approx(0.2f));
    CHECK(r[2] == doctest::Approx(0.3f));
    CHECK(r[3] == doctest::Approx(0.0f));

    auto mm = m.matmul(Tensor::transpose(m));
    CHECK(mm[0] == doctest::Approx(0.05f));
    CHECK(mm[1] == doctest::Approx(0.01f));
    CHECK(mm[2] == doctest::Approx(0.01f));
    CHECK(mm[3] == doctest::Approx(0.1f));
}

/// @brief Loss parity with PyTorch
TEST_CASE("mse loss parity") {
    Tensor pred({1,2});
    pred[0]=1.0f; pred[1]=2.0f;
    Tensor target({1,2});
    target.fill(0.0f);

    MSELoss loss;
    float l = loss(pred, target);
    CHECK(l == doctest::Approx(2.5f));
    pred = Tensor({1,2},0.0f,true);
    pred[0]=1.0f; pred[1]=2.0f;
    l = loss(pred, target);
    loss.backward(pred, target);
    CHECK(pred.grad()[0] == doctest::Approx(1.0f));
    CHECK(pred.grad()[1] == doctest::Approx(2.0f));
}

/// @brief Elementwise operations parity
TEST_CASE("elementwise parity") {
    Tensor a({2,2});
    a[0]=1.0f; a[1]=2.0f; a[2]=3.0f; a[3]=4.0f;
    Tensor b({2,2});
    b[0]=4.0f; b[1]=3.0f; b[2]=2.0f; b[3]=1.0f;

    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;

    CHECK(add[0] == doctest::Approx(5.0f));
    CHECK(add[1] == doctest::Approx(5.0f));
    CHECK(add[2] == doctest::Approx(5.0f));
    CHECK(add[3] == doctest::Approx(5.0f));

    CHECK(sub[0] == doctest::Approx(-3.0f));
    CHECK(sub[1] == doctest::Approx(-1.0f));
    CHECK(sub[2] == doctest::Approx(1.0f));
    CHECK(sub[3] == doctest::Approx(3.0f));

    CHECK(mul[0] == doctest::Approx(4.0f));
    CHECK(mul[1] == doctest::Approx(6.0f));
    CHECK(mul[2] == doctest::Approx(6.0f));
    CHECK(mul[3] == doctest::Approx(4.0f));
}

