#include <doctest/doctest.h>
#include "mini_torch/embedding.h"
#include <vector>
#include <concepts>

/// @brief Verify embedding lookup by indices
TEST_CASE("embedding lookup") {
    Embedding emb(4, 2);
    auto &w = emb.weight();
    for(size_t i=0;i<w.size();++i)
        w[i] = static_cast<float>(i);

    std::vector<size_t> idx = {1, 3};
    Tensor out = emb(idx);
    CHECK(out.shape() == std::vector<size_t>{2,2});
    CHECK(out[0] == doctest::Approx(2.0f));
    CHECK(out[1] == doctest::Approx(3.0f));
    CHECK(out[2] == doctest::Approx(6.0f));
    CHECK(out[3] == doctest::Approx(7.0f));
}

static_assert(std::invocable<Embedding, const std::vector<size_t>&>);
