#pragma once
#include <memory>
#include <functional>
#include "tensor.h"

/// @brief Base autograd function node
class Function {
public:
    /// @brief Virtual destructor
    virtual ~Function() = default;
    /// @brief Backward pass given gradient of outputs
    virtual void backward(const Tensor &grad_output) = 0;
};

/// @brief Addition backward computation
class AddFunction : public Function {
public:
    /// @brief Construct from input tensors
    AddFunction(Tensor &a, Tensor &b);
    /// @brief Propagate gradients to inputs
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< left operand
    std::reference_wrapper<Tensor> m_b; ///< right operand
    Tensor m_a_saved; ///< copy of left operand
    Tensor m_b_saved; ///< copy of right operand
};

/// @brief Subtraction backward computation
class SubFunction : public Function {
public:
    /// @brief Construct from input tensors
    SubFunction(Tensor &a, Tensor &b);
    /// @brief Propagate gradients to inputs
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< minuend
    std::reference_wrapper<Tensor> m_b; ///< subtrahend
    Tensor m_a_saved; ///< copy of minuend
    Tensor m_b_saved; ///< copy of subtrahend
};

/// @brief Multiplication backward computation
class MulFunction : public Function {
public:
    /// @brief Construct from input tensors
    MulFunction(Tensor &a, Tensor &b);
    /// @brief Propagate gradients to inputs
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< left operand
    std::reference_wrapper<Tensor> m_b; ///< right operand
    Tensor m_a_saved; ///< copy of left operand
    Tensor m_b_saved; ///< copy of right operand
};

/// @brief Matrix multiplication backward computation
class MatMulFunction : public Function {
public:
    /// @brief Construct from input tensors and save copies for backward
    MatMulFunction(Tensor &a, Tensor &b);
    /// @brief Propagate gradients to inputs
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< left matrix
    std::reference_wrapper<Tensor> m_b; ///< right matrix
    Tensor m_a_saved;   ///< copy of left matrix
    Tensor m_b_saved;   ///< copy of right matrix
};

/// @brief ReLU backward computation
class ReLUFunction : public Function {
public:
    /// @brief Construct from input tensor and mask
    ReLUFunction(Tensor &a, Tensor mask);
    /// @brief Propagate gradients to input
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< input tensor
    Tensor m_mask; ///< mask of activated values
};

/// @brief Transpose backward computation
class TransposeFunction : public Function {
public:
    /// @brief Construct from input tensor
    explicit TransposeFunction(Tensor &a);
    /// @brief Propagate gradients to input
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< input tensor
};

/// @brief Softmax backward computation
class SoftmaxFunction : public Function {
public:
    /// @brief Construct from input tensor and output
    SoftmaxFunction(Tensor &a, Tensor output);
    /// @brief Propagate gradients to input
    void backward(const Tensor &grad_output) override;
private:
    std::reference_wrapper<Tensor> m_a; ///< input tensor
    Tensor m_output;  ///< cached softmax output
};
