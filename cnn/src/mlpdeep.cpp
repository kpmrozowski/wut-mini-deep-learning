// Copyright 2020-present pytorch-cpp Authors
#include <mlpdeep.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/dropout.h>
#include <torch/torch.h>

using torch::nn::Sigmoid;
using torch::nn::Dropout;
using torch::nn::DropoutOptions;

Sequential MlpDeepImpl::layer0{
    Linear(3 * 32 * 32, 1024),
    Dropout(DropoutOptions(0.2)),
    Sigmoid(),
};
Sequential MlpDeepImpl::layer1{
    Linear(1024, 256),
    Dropout(DropoutOptions(0.5)),
    Sigmoid(),
};
Sequential MlpDeepImpl::layer2{
    Linear(256, 64),
    Dropout(DropoutOptions(0.5)),
    Sigmoid(),
};

MlpDeepImpl::MlpDeepImpl(int64_t num_classes)
 : fc3(64, num_classes) {
    register_module("layer0", layer0);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("fc3", fc3);
}

torch::Tensor MlpDeepImpl::forward(torch::Tensor x) {
    x = x.view({-1,  3 * 32 * 32});
    x = layer0->forward(x);
    x = layer1->forward(x);
    x = layer2->forward(x);
    return fc3->forward(x);
}
