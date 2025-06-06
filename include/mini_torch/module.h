#pragma once
#include <vector>

/// @brief Base class mirroring torch::nn::Module
class Module {
public:
    /// @brief Virtual destructor
    virtual ~Module() = default;
    /// @brief Retrieve parameters requiring optimization
    virtual std::vector<class Tensor*> parameters() = 0;
};
