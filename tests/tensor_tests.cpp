#include "mini_torch/tensor.h"
#include <catch_amalgamated.hpp>
#include <concepts>

TEST_CASE("tensor add", "[tensor]") {
    Tensor t({2, 2}, 1.0f);
    t[0] = 2.0f;
    Tensor t2({2, 2}, 3.0f);
    auto sum = Tensor::add(t, t2);
    REQUIRE(sum.size() == 4);
    REQUIRE(sum[0] == 5.0f);
}

TEST_CASE("tensor matmul", "[tensor]") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    float vals_a[] = {1,2,3,4,5,6};
    float vals_b[] = {7,8,9,10,11,12};
    for (size_t i=0;i<6;++i) a[i]=vals_a[i];
    for (size_t i=0;i<6;++i) b[i]=vals_b[i];
    auto out = Tensor::matmul(a,b);
    REQUIRE(out.shape() == std::vector<size_t>{2,2});
    REQUIRE(out[0] == 58.0f);
    REQUIRE(out[1] == 64.0f);
    REQUIRE(out[2] == 139.0f);
    REQUIRE(out[3] == 154.0f);
}

TEST_CASE("tensor relu", "[tensor]") {
    Tensor t({1,4});
    float vals[] = {-1.0f,0.0f,2.0f,-3.0f};
    for(size_t i=0;i<4;++i) t[i]=vals[i];
    t.relu();
    REQUIRE(t[0] == 0.0f);
    REQUIRE(t[1] == 0.0f);
    REQUIRE(t[2] == 2.0f);
    REQUIRE(t[3] == 0.0f);
}

static_assert(TensorLike<Tensor>);

