#include "mini_torch/model.h"
#include "mini_torch/loss.h"
#include "mini_torch/optim.h"

Model::Model(size_t dim)
    : m_proj_q(dim, dim), m_proj_k(dim, dim), m_proj_v(dim, dim), m_out(dim, dim) {}

Tensor Model::operator()(const Tensor &input) const {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = Attention::apply(q, k, v);
    return m_out(attn);
}

void Model::train_step(const Tensor &input, const Tensor &target, float lr) {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = Attention::apply(q, k, v);
    auto out = m_out(attn);
    MSELoss loss;
    loss(out, target);
    loss.backward(out, target);
    Tensor grad_out = out.grad();
    Tensor grad_w(m_out.weight().shape());
    Tensor grad_b(m_out.bias().shape());
    size_t batch = attn.shape()[0];
    size_t in_dim = m_out.weight().shape()[0];
    size_t out_dim = m_out.weight().shape()[1];
    for(size_t b =0;b<batch;++b){
        for(size_t i=0;i<in_dim;++i){
            for(size_t j=0;j<out_dim;++j){
                grad_w[i*out_dim+j] += attn[b*in_dim+i]*grad_out[b*out_dim+j];
            }
        }
    }
    for(size_t j=0;j<out_dim;++j){
        for(size_t b=0;b<batch;++b)
            grad_b[j]+=grad_out[b*out_dim+j];
    }
    m_out.weight().grad() = grad_w;
    m_out.bias().grad() = grad_b;
    SGD opt({&m_out.weight(), &m_out.bias()}, lr);
    opt.step();
}

GenesisModel::GenesisModel(size_t dim)
    : m_proj_q(dim, dim), m_proj_k(dim, dim), m_proj_v(dim, dim), m_out(dim, dim), m_attn(8, dim) {}

Tensor GenesisModel::operator()(const Tensor &input) const {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = m_attn(q, k, v);
    return m_out(attn);
}

void GenesisModel::train_step(const Tensor &input, const Tensor &target, float lr) {
    auto q = m_proj_q(input);
    auto k = m_proj_k(input);
    auto v = m_proj_v(input);
    auto attn = m_attn(q, k, v);
    auto out = m_out(attn);
    MSELoss loss;
    loss(out, target);
    loss.backward(out, target);
    Tensor grad_out = out.grad();
    Tensor grad_w(m_out.weight().shape());
    Tensor grad_b(m_out.bias().shape());
    size_t batch = attn.shape()[0];
    size_t in_dim = m_out.weight().shape()[0];
    size_t out_dim = m_out.weight().shape()[1];
    for(size_t b_i=0;b_i<batch;++b_i){
        for(size_t i=0;i<in_dim;++i){
            for(size_t j=0;j<out_dim;++j){
                grad_w[i*out_dim+j]+=attn[b_i*in_dim+i]*grad_out[b_i*out_dim+j];
            }
        }
    }
    for(size_t j=0;j<out_dim;++j){
        for(size_t b_i=0;b_i<batch;++b_i)
            grad_b[j]+=grad_out[b_i*out_dim+j];
    }
    m_out.weight().grad() = grad_w;
    m_out.bias().grad() = grad_b;
    SGD opt({&m_out.weight(), &m_out.bias()}, lr);
    opt.step();
}

std::vector<Tensor*> Model::parameters() {
    auto p = m_proj_q.parameters();
    auto pk = m_proj_k.parameters();
    auto pv = m_proj_v.parameters();
    auto po = m_out.parameters();
    p.insert(p.end(), pk.begin(), pk.end());
    p.insert(p.end(), pv.begin(), pv.end());
    p.insert(p.end(), po.begin(), po.end());
    return p;
}

std::vector<Tensor*> GenesisModel::parameters() {
    auto p = m_proj_q.parameters();
    auto pk = m_proj_k.parameters();
    auto pv = m_proj_v.parameters();
    auto po = m_out.parameters();
    p.insert(p.end(), pk.begin(), pk.end());
    p.insert(p.end(), pv.begin(), pv.end());
    p.insert(p.end(), po.begin(), po.end());
    auto ap = m_attn.parameters();
    p.insert(p.end(), ap.begin(), ap.end());
    return p;
}
