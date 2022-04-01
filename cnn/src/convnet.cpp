// Copyright 2020-present pytorch-cpp Authors
#include <convnet.h>
#include <torch/torch.h>

ConvNetImpl::ConvNetImpl(int64_t num_classes)
    : fc1(256 * 4 * 4, 128), fc2(128, num_classes) {
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
    register_module("fc1", fc1);
    register_module("fc2", fc2);
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
    x = fc1->forward(x);
    return fc2->forward(x);
}

void ConvNetImpl::print_modules() {
    auto modules = this->modules();
    long params_size = 0;
    for (auto module : modules) {
        module->pretty_print(std::cout);
        std::cout << std::endl;
        auto dict = module->named_parameters(false);
        for (auto record : dict) {
            auto name = record.key();
            auto param = record.value();
            long module_size = 1;
            std::cout << name << ":\tdim=(";
            for (long size : param.sizes()) {
                module_size *= size;
                std::cout << size << ",";
            }
            std::cout << "), so module parameters count is " << module_size << std::endl;
            params_size += module_size;
            // std::cout << "NAME:\t" << name << "\nPARAMETERS:\n" << param << std::endl << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "all parameters count is " << params_size << std::endl;
}
