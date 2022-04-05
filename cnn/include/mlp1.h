// Copyright 2020-present pytorch-cpp Authors
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

class Mlp1Impl : public torch::nn::Module {
 public:
    Mlp1Impl() = delete;
    explicit Mlp1Impl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    Sequential layer;
    int64_t m_num_classes;
};

TORCH_MODULE(Mlp1);
