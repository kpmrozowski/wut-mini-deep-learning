#pragma once

#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/dropout.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/padding.h>
#include <torch/nn/options/dropout.h>
#include <torch/torch.h>

using torch::nn::Sequential;
using torch::nn::Linear;

class MlpDeepImpl : public torch::nn::Module {
 public:
    MlpDeepImpl() = delete;
    explicit MlpDeepImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    static Sequential layer0;
    static Sequential layer1;
    static Sequential layer2;
    Linear fc3;
};

TORCH_MODULE(MlpDeep);
