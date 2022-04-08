#pragma once

#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/dropout.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/padding.h>
#include <torch/nn/options/dropout.h>
#include <torch/torch.h>

class MlpDeepImpl : public torch::nn::Module {
 public:
    MlpDeepImpl() = delete;
    explicit MlpDeepImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Linear fc0;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

TORCH_MODULE(MlpDeep);
