#include <doctest/doctest.h>
#include "mini_torch/data.h"

/// @brief Verify dataset iteration and dataloader batching
TEST_CASE("dataset and dataloader") {
    std::vector<int> samples = {1,2,3,4,5};
    Dataset<int> ds(samples);
    CHECK(ds.size() == 5);
    CHECK(ds[2] == 3);

    DataLoader loader(ds, 2);
    std::vector<int> out;
    for(auto batch : loader){
        for(int v : batch) out.push_back(v);
    }
    CHECK(out == samples);
}

/// @brief Ensure text dataset loads characters correctly
TEST_CASE("load tiny shakespeare") {
    auto ds = load_text_dataset("../data/tinyshakespeare.txt");
    CHECK(ds.size() > 0);
    CHECK(ds[0] == 'F');
    DataLoader loader(ds, 8);
    size_t count = 0;
    for(auto b : loader) count += b.size();
    CHECK(count == ds.size());
}

static_assert(std::ranges::range<Dataset<int>>);
static_assert(std::ranges::input_range<DataLoader<Dataset<int>>>);
