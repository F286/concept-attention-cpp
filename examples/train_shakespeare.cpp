#include "mini_torch/embedding.h"
#include "mini_torch/model.h"
#include "mini_torch/loss.h"
#include "mini_torch/optim.h"
#include "mini_torch/data.h"
#include <iostream>
#include <vector>
#include <random>

int main(){
    const size_t block = 8;
    const size_t dim = 32;
    const size_t vocab = 256;
    const float lr = 0.05f;
    const int epochs = 10;

    auto dataset = load_text_dataset("../data/tinyshakespeare.txt");
    size_t limit = std::min<size_t>(1024, dataset.size());
    Embedding embed(vocab, dim);
    Model transformer(dim);
    Linear head(dim, vocab);

    std::vector<Tensor*> params;
    auto mp = transformer.parameters();
    params.insert(params.end(), mp.begin(), mp.end());
    auto hp = head.parameters();
    params.insert(params.end(), hp.begin(), hp.end());
    params.push_back(&embed.weight());
    SGD opt(params, lr);

    CrossEntropyLoss loss;
    for(int epoch=0; epoch<epochs; ++epoch){
        float total=0.0f; size_t count=0;
        for(size_t i=0;i+block<limit;i+=block){
            std::vector<size_t> idx(block);
            std::vector<size_t> tgt(block);
            for(size_t j=0;j<block;++j){
                idx[j] = static_cast<unsigned char>(dataset[i+j]);
                tgt[j] = static_cast<unsigned char>(dataset[i+j+1]);
            }
            auto x = embed(idx);
            auto out = transformer(x);
            auto logits = head(out);
            float l = loss(logits, tgt);
            loss.backward(logits, tgt);
            logits.backward(logits.grad());
            opt.step();
            opt.zero_grad();
            total += l; ++count;
        }
        std::cout << "Epoch " << epoch << " loss " << total / static_cast<float>(count) << '\n';
    }

    // generate 32 characters starting from first block
    std::vector<size_t> ctx(block);
    for(size_t j=0;j<block;++j) ctx[j] = static_cast<unsigned char>(dataset[j]);
    std::cout << "Generated:\n";
    for(int step=0; step<32; ++step){
        auto x = embed(ctx);
        auto out = transformer(x);
        auto logits = head(out);
        size_t last = ctx.back();
        // pick argmax at last position
        size_t seq_pos = block-1;
        size_t pred=0; float maxv=logits.at(seq_pos,0);
        for(size_t i=1;i<vocab;++i){
            float v = logits.at(seq_pos,i);
            if(v>maxv){ maxv=v; pred=i; }
        }
        ctx.erase(ctx.begin()); ctx.push_back(pred);
        std::cout << static_cast<char>(pred);
    }
    std::cout << "\n";
    return 0;
}
