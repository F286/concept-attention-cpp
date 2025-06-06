#include "mini_torch/model.h"
#include <iostream>
#include <random>

/**
 * @brief Run tiny training loop comparing baseline and genesis models
 */
int run_training_demo() {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Tensor input({4,4});
    Tensor target({4,4});
    for (size_t i=0;i<input.size();++i) input[i]=dist(rng);
    for (size_t i=0;i<target.size();++i) target[i]=dist(rng);

    Model baseline(4);
    GenesisModel genesis(4);

    const float lr = 0.1f;
    for(int epoch=0; epoch<5; ++epoch){
        baseline.train_step(input, target, lr);
        genesis.train_step(input, target, lr);
        auto out_b = baseline(input);
        auto out_g = genesis(input);
        float loss_b=0.0f, loss_g=0.0f;
        for(size_t i=0;i<out_b.size();++i){
            loss_b += 0.5f*(out_b[i]-target[i])*(out_b[i]-target[i]);
            loss_g += 0.5f*(out_g[i]-target[i])*(out_g[i]-target[i]);
        }
        std::cout << "epoch " << epoch << ": baseline " << loss_b
                  << ", genesis " << loss_g << "\n";
    }
    return 0;
}

