#pragma once
#include <string>
#include <vector>

/// @brief Basic English tokenizer splitting on punctuation and whitespace
class Tokenizer {
public:
    /// @brief Split text into lowercase tokens
    std::vector<std::string> operator()(const std::string &text) const;
};
