// Copyright 2020-present pytorch-cpp Authors
#include "convnet.h"
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes)
    : fc(256 * 4 * 4, num_classes) {
    register_module("layer010", layer01);
    register_module("layer020", layer02);
    register_module("layer030", layer03);
    register_module("layer040", layer04);
    register_module("layer050", layer05);
   //  register_module("layer051", layer05);
   //  register_module("layer060", layer06);
   //  register_module("layer070", layer07);
   //  register_module("layer071", layer07);
   //  register_module("layer072", layer07);
   //  register_module("layer080", layer08);
   //  register_module("layer090", layer09);
   //  register_module("layer091", layer09);
   //  register_module("layer092", layer09);
   //  register_module("layer100", layer10);
   //  register_module("layer110", layer11);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
    x = layer01->forward(x);
    x = layer02->forward(x);
    x = layer03->forward(x);
    x = layer04->forward(x);
    x = layer05->forward(x);
   //  x = layer05->forward(x);
   //  x = layer06->forward(x);
   //  x = layer07->forward(x);
   //  x = layer07->forward(x);
   //  x = layer07->forward(x);
   //  x = layer08->forward(x);
   //  x = layer09->forward(x);
   //  x = layer09->forward(x);
   //  x = layer09->forward(x);
   //  x = layer10->forward(x);
   //  x = layer11->forward(x);
    x = pool->forward(x);
    x = x.view({-1,  256 * 4 * 4});
    return fc->forward(x);
}
