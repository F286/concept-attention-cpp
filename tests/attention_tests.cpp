#include <doctest/doctest.h>
#include "mini_torch/attention.h"

/// @brief Verify standard attention output
TEST_CASE("standard attention") {
    Tensor q({1, 2});
    Tensor k({2, 2});
    Tensor v({2, 2});
    q[0] = 1.0f; q[1] = 0.0f;
    k[0] = 1.0f; k[1] = 0.0f; k[2] = 0.0f; k[3] = 1.0f;
    v[0] = 2.0f; v[1] = 3.0f; v[2] = 4.0f; v[3] = 5.0f;
    auto out = Attention::apply(q, k, v);
    CHECK(out.shape() == std::vector<size_t>{1, 2});
    CHECK(out[0] == doctest::Approx(2.6604769f));
    CHECK(out[1] == doctest::Approx(3.6604769f));
}

/// @brief Verify genesis attention shape
TEST_CASE("genesis attention") {
    Tensor q({2, 4}, 0.5f);
    Tensor k({2, 4}, 0.5f);
    Tensor v({2, 4}, 1.0f);
    GenesisAttention ga(8, 2);
    auto out = ga(q, k, v);
    CHECK(out.shape() == std::vector<size_t>{2, 4});
}
