// Copyright 2020-present pytorch-cpp Authors
#include <mlp1.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>


Mlp1Impl::Mlp1Impl(int64_t num_classes)
    : fc(3 * 32 * 32, num_classes) {
    register_module("fc", fc);
}

torch::Tensor Mlp1Impl::forward(torch::Tensor x) {
    x = x.view({-1,  3 * 32 * 32});
    return fc->forward(x);
}
