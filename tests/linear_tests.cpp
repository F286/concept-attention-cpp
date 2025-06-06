#include <doctest/doctest.h>
#include "mini_torch/linear.h"
#include "mini_torch/optim.h"

TEST_CASE("linear forward backward single") {
    Linear layer(2,2);
    auto &w = layer.weight();
    w[0] = 1.0f; w[1] = 2.0f; w[2] = 3.0f; w[3] = 4.0f;
    auto &b = layer.bias();
    b[0] = 0.5f; b[1] = -0.5f;

    Tensor input({1,2});
    input[0] = 1.0f; input[1] = 2.0f;
    auto out = layer(input);
    CHECK(out.shape() == std::vector<size_t>{1,2});
    CHECK(out[0] == doctest::Approx(7.5f));
    CHECK(out[1] == doctest::Approx(9.5f));

    Tensor target({1,2});
    target.fill(0.0f);
    Tensor grad(out.shape());
    for(size_t i=0;i<out.size();++i) grad[i] = out[i] - target[i];

    Tensor grad_w(layer.weight().shape());
    Tensor grad_b(layer.bias().shape());
    for(size_t i=0;i<2;++i){
        grad_w[i] = input[0] * grad[i];
        grad_w[2+i] = input[1] * grad[i];
        grad_b[i] = grad[i];
    }

    layer.weight().grad() = grad_w;
    layer.bias().grad() = grad_b;
    SGD opt({&layer.weight(), &layer.bias()}, 0.1f);
    opt.step();

    CHECK(layer.weight()[0] == doctest::Approx(0.25f));
    CHECK(layer.weight()[1] == doctest::Approx(1.05f));
    CHECK(layer.weight()[2] == doctest::Approx(1.5f));
    CHECK(layer.weight()[3] == doctest::Approx(2.1f));
}

TEST_CASE("linear batch step") {
    Linear layer(2,1);
    auto &w = layer.weight();
    w.fill(0.0f);
    auto &b = layer.bias();
    b.fill(0.0f);

    Tensor input({2,2});
    input[0]=1.0f; input[1]=2.0f;
    input[2]=3.0f; input[3]=4.0f;
    Tensor out = layer(input);
    CHECK(out[0] == doctest::Approx(0.0f));
    CHECK(out[1] == doctest::Approx(0.0f));

    Tensor target({2,1});
    target[0]=1.0f; target[1]=1.0f;
    Tensor grad(out.shape());
    for(size_t i=0;i<out.size();++i) grad[i] = out[i] - target[i];

    Tensor grad_w(layer.weight().shape());
    Tensor grad_b(layer.bias().shape());
    for(size_t b_i=0;b_i<2;++b_i){
        for(size_t j=0;j<2;++j) {
            grad_w[j] += input[b_i*2 + j] * grad[b_i];
        }
        grad_b[0] += grad[b_i];
    }
    layer.weight().grad() = grad_w;
    layer.bias().grad() = grad_b;
    SGD opt({&layer.weight(), &layer.bias()}, 0.5f);
    opt.step();

    CHECK(layer.weight()[0] == doctest::Approx(2.0f));
    CHECK(layer.weight()[1] == doctest::Approx(3.0f));
    CHECK(layer.bias()[0] == doctest::Approx(1.0f));
}
