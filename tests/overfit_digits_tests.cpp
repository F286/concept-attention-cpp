#include <doctest/doctest.h>
#include "mini_torch/model.h"
#include "mini_torch/data.h"

/// @brief Training sample of input and target tensors
struct Sample {
    Tensor input;  ///< input tensor
    Tensor target; ///< target tensor
};

/// @brief Overfit baseline model on digits
TEST_CASE("[TRAIN] overfit digits") {
    std::vector<Sample> samples;
    for(size_t i=0;i<9;++i){
        Tensor in({1,10}, 0.0f);
        in[i] = 1.0f;
        Tensor tgt({1,10}, 0.0f);
        tgt[i+1] = 1.0f;
        samples.push_back({in,tgt});
    }
    Dataset<Sample> ds(samples);
    DataLoader loader(ds, 1, false);

    Model model(10);
    const float lr = 0.5f;

    for(int epoch=0; epoch<200; ++epoch){
        for(auto batch : loader){
            const Sample &s = batch[0];
            model.train_step(s.input, s.target, lr);
        }
    }

    for(size_t i=0;i<9;++i){
        Tensor in({1,10}, 0.0f);
        in[i] = 1.0f;
        auto out = model(in);
        size_t pred = 0;
        float max_v = out[0];
        for(size_t j=1;j<10;++j){
            if(out[j] > max_v){ max_v = out[j]; pred = j; }
        }
        CHECK(pred == i+1);
    }
}

