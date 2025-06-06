#pragma once
#include "tensor.h"
#include "module.h"
#include <vector>
#include <random>

/// @brief Randomly zeroes elements during training
class Dropout : public Module {
public:
    /// @brief Construct with drop probability
    explicit Dropout(float p);
    /// @brief Apply dropout
    Tensor operator()(const Tensor &input) const;
    /// @brief Set training mode
    void train(bool mode);
    /// @brief Parameters list (none)
    std::vector<Tensor*> parameters() override;
private:
    float m_p; ///< drop probability
    bool m_training; ///< training flag
    mutable std::mt19937 m_rng; ///< random generator
};
