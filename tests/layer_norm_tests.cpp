#include <doctest/doctest.h>
#include "mini_torch/layer_norm.h"
#include "mini_torch/tensor.h"
#include <concepts>

/// @brief verify normalization for single sample
TEST_CASE("layernorm basic") {
    LayerNorm ln(2);
    Tensor x({1,2});
    x[0] = 1.0f; x[1] = 3.0f;
    Tensor out = ln(x);
    float mean = (1.0f + 3.0f)/2.0f;
    float var = ((1.0f-mean)*(1.0f-mean)+(3.0f-mean)*(3.0f-mean))/2.0f;
    float denom = 1.0f/std::sqrt(var + 1e-5f);
    CHECK(out[0] == doctest::Approx((1.0f-mean)*denom));
    CHECK(out[1] == doctest::Approx((3.0f-mean)*denom));
}

/// @brief verify parameters influence output
TEST_CASE("layernorm affine") {
    LayerNorm ln(2);
    ln.weight()[0] = 2.0f; ln.weight()[1] = 3.0f;
    ln.bias()[0] = 1.0f; ln.bias()[1] = -1.0f;
    Tensor x({1,2});
    x[0] = 1.0f; x[1] = 3.0f;
    Tensor out = ln(x);
    float mean = (1.0f + 3.0f)/2.0f;
    float var = ((1.0f-mean)*(1.0f-mean)+(3.0f-mean)*(3.0f-mean))/2.0f;
    float denom = 1.0f/std::sqrt(var + 1e-5f);
    CHECK(out[0] == doctest::Approx(((1.0f-mean)*denom)*2.0f + 1.0f));
    CHECK(out[1] == doctest::Approx(((3.0f-mean)*denom)*3.0f - 1.0f));
}

/// @brief verify per-sample normalization for batch input
TEST_CASE("layernorm batch") {
    LayerNorm ln(2);
    Tensor x({2,2});
    x[0] = 1.0f; x[1] = 3.0f;
    x[2] = 0.0f; x[3] = 4.0f;
    Tensor out = ln(x);
    // first sample
    float mean0 = (1.0f + 3.0f)/2.0f;
    float var0 = ((1.0f-mean0)*(1.0f-mean0)+(3.0f-mean0)*(3.0f-mean0))/2.0f;
    float denom0 = 1.0f/std::sqrt(var0 + 1e-5f);
    CHECK(out[0] == doctest::Approx((1.0f-mean0)*denom0));
    CHECK(out[1] == doctest::Approx((3.0f-mean0)*denom0));
    // second sample
    float mean1 = (0.0f + 4.0f)/2.0f;
    float var1 = ((0.0f-mean1)*(0.0f-mean1)+(4.0f-mean1)*(4.0f-mean1))/2.0f;
    float denom1 = 1.0f/std::sqrt(var1 + 1e-5f);
    CHECK(out[2] == doctest::Approx((0.0f-mean1)*denom1));
    CHECK(out[3] == doctest::Approx((4.0f-mean1)*denom1));
}

static_assert(std::derived_from<LayerNorm, Module>);
