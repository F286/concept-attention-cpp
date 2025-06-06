#include <doctest/doctest.h>
#include "mini_torch/tokenizer.h"
#include <concepts>

/// @brief verify lowercase splitting on punctuation
TEST_CASE("tokenizer basic") {
    Tokenizer tok;
    auto out = tok("Hello, World!");
    REQUIRE(out.size() == 2);
    CHECK(out[0] == "hello");
    CHECK(out[1] == "world");
}

/// @brief handle numbers and repeated punctuation
TEST_CASE("tokenizer punctuation and numbers") {
    Tokenizer tok;
    auto out = tok("Numbers 123 and punctuation!!!");
    std::vector<std::string> ref{"numbers","123","and","punctuation"};
    CHECK(out == ref);
}

/// @brief handle whitespace variations
TEST_CASE("tokenizer whitespace") {
    Tokenizer tok;
    auto out = tok("Hello   world\nagain");
    std::vector<std::string> ref{"hello","world","again"};
    CHECK(out == ref);
}

static_assert(std::invocable<Tokenizer,const std::string&>);
