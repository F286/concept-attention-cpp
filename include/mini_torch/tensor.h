#pragma once
#include <vector>
#include <cassert>
#include <concepts>
#include <experimental/simd>
#include <memory>

class Function;

using floatv = std::experimental::native_simd<float>;

/// @brief Proxy reference to a float inside a SIMD chunk
struct ScalarRef {
    floatv* chunk; ///< owning SIMD chunk
    size_t lane;   ///< lane index
    /// @brief Convert to float
    operator float() const { return (*chunk)[lane]; }
    /// @brief Assign value
    ScalarRef& operator=(float v) { (*chunk)[lane] = v; return *this; }
    /// @brief Add value
    ScalarRef& operator+=(float v) { (*chunk)[lane] += v; return *this; }
    /// @brief Subtract value
    ScalarRef& operator-=(float v) { (*chunk)[lane] -= v; return *this; }
    /// @brief Multiply value
    ScalarRef& operator*=(float v) { (*chunk)[lane] *= v; return *this; }
    /// @brief Divide value
    ScalarRef& operator/=(float v) { (*chunk)[lane] /= v; return *this; }
};

/**
 * @brief A minimal tensor container mimicking a subset of std::vector for numeric data.
 * It stores elements in row major order and supports basic arithmetic and matrix
 * operations used for small neural network prototypes.
 */
class Tensor {
public:
    /// @brief Default construct empty tensor
    Tensor() = default;
    /// @brief Copy construct tensor
    Tensor(const Tensor& other);
    /// @brief Copy assign tensor
    Tensor& operator=(const Tensor& other);
    /// @brief Construct tensor with shape and initial value
    Tensor(std::vector<size_t> shape, float value = 0.0f, bool requires_grad = false);
    /// @brief Access element by flat index
    ScalarRef operator[](size_t idx);
    /// @brief Const access by flat index
    float operator[](size_t idx) const;
    /// @brief Access element by row and column in 2D tensor
    ScalarRef at(size_t row, size_t col);
    /// @brief Const access by row and column in 2D tensor
    float at(size_t row, size_t col) const;
    /// @brief Iterator to first element
    auto begin() { return m_data.begin(); }
    /// @brief Const iterator to first element
    auto begin() const { return m_data.begin(); }
    /// @brief Iterator past last element
    auto end() { return m_data.end(); }
    /// @brief Const iterator past last element
    auto end() const { return m_data.end(); }
    /// @brief Return underlying data size
    size_t size() const;
    /// @brief Return tensor shape
    const std::vector<size_t> &shape() const;
    /// @brief Matrix multiplication
    static Tensor matmul(const Tensor &a, const Tensor &b);
    /// @brief Elementwise addition
    static Tensor add(const Tensor &a, const Tensor &b);
    /// @brief Elementwise subtraction
    static Tensor sub(const Tensor &a, const Tensor &b);
    /// @brief Elementwise multiplication
    static Tensor mul(const Tensor &a, const Tensor &b);
    /// @brief Matrix multiplication
    Tensor matmul(const Tensor &other) const;
    /// @brief Elementwise addition operator
    Tensor operator+(const Tensor &other) const;
    /// @brief Elementwise subtraction operator
    Tensor operator-(const Tensor &other) const;
    /// @brief Elementwise multiplication operator
    Tensor operator*(const Tensor &other) const;
    /// @brief Transpose 2D tensor
    static Tensor transpose(const Tensor &t);
    /// @brief Row-wise softmax for a 2D tensor
    static Tensor softmax(const Tensor &t);
    /// @brief Out of place ReLU
    Tensor relu() const;
    /// @brief In place ReLU
    void relu_();
    /// @brief Fill tensor with value
    void fill(float v);
    /// @brief Access gradient tensor
    Tensor &grad();
    /// @brief Const access gradient tensor
    const Tensor &grad() const;
    /// @brief Whether tensor requires gradients
    bool requires_grad() const;
    /// @brief Zero stored gradients
    void zero_grad();
    /// @brief Backpropagate using unit gradient
    void backward();
    /// @brief Backpropagate with supplied gradient
    void backward(const Tensor &grad_output);

private:
    std::vector<size_t> m_shape; ///< tensor dimensions
    size_t m_size{};             ///< scalar element count
    std::vector<floatv> m_data;  ///< SIMD element storage
    mutable std::unique_ptr<Tensor> m_grad; ///< stored gradient
    bool m_requires_grad{};      ///< autograd flag
    std::shared_ptr<Function> m_grad_fn; ///< backward function
};

/// @brief Concept requirement similar to std::ranges::range
template<typename T>
concept TensorLike = requires(T t) {
    { t.size() } -> std::same_as<size_t>;
    { t.shape() } -> std::same_as<const std::vector<size_t>&>;
};
