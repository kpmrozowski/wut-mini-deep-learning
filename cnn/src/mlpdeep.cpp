// Copyright 2020-present pytorch-cpp Authors
#include <mlpdeep.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>


MlpDeepImpl::MlpDeepImpl(int64_t num_classes)
    : fc0(3 * 32 * 32, 1024),
      fc1(1024, 256),
      fc2(256, 64),
      fc3(64, num_classes) {
    register_module("fc0", fc0);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor MlpDeepImpl::forward(torch::Tensor x) {
    x = x.view({-1,  3 * 32 * 32});
    x = fc0->forward(x);
    x = fc1->forward(x);
    x = fc2->forward(x);
    return fc3->forward(x);
}
