#pragma once
#include <vector>
#include <cassert>
#include <concepts>

/**
 * @brief A minimal tensor container mimicking a subset of std::vector for numeric data.
 * It stores elements in row major order and supports basic arithmetic and matrix
 * operations used for small neural network prototypes.
 */
class Tensor {
public:
    /// @brief Construct tensor with shape and initial value
    Tensor(std::vector<size_t> shape, float value = 0.0f);
    /// @brief Access element by flat index
    float &operator[](size_t idx);
    /// @brief Const access by flat index
    const float &operator[](size_t idx) const;
    /// @brief Return underlying data size
    size_t size() const;
    /// @brief Return tensor shape
    const std::vector<size_t> &shape() const;
    /// @brief Matrix multiplication
    static Tensor matmul(const Tensor &a, const Tensor &b);
    /// @brief Elementwise addition
    static Tensor add(const Tensor &a, const Tensor &b);
    /// @brief Apply ReLU
    void relu();

private:
    std::vector<size_t> m_shape; ///< tensor dimensions
    std::vector<float> m_data;   ///< element storage
};

/// @brief Concept requirement similar to std::ranges::range
template<typename T>
concept TensorLike = requires(T t) {
    { t.size() } -> std::same_as<size_t>;
    { t.shape() } -> std::same_as<const std::vector<size_t>&>;
};
