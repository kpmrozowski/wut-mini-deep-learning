// Copyright 2020-present pytorch-cpp Authors
#include <mlp1.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>

using torch::nn::Linear;
using torch::nn::Sigmoid;
using torch::nn::Dropout;
using torch::nn::DropoutOptions;

Mlp1Impl::Mlp1Impl(int64_t num_classes)
    : layer(
        Sequential(
            Linear(3 * 32 * 32, num_classes),
            Dropout(DropoutOptions(0.2)),
            Sigmoid()
        )
    )
{
    register_module("layer", layer);
}

torch::Tensor Mlp1Impl::forward(torch::Tensor x) {
    x = x.view({-1,  3 * 32 * 32});
    return layer->forward(x);
}
