// Copyright 2020-present pytorch-cpp Authors
#include <convnet.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>

torch::nn::Sequential ConvNetImpl::layer01{
    torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).bias(false)),
    torch::nn::BatchNorm2d(16),
    torch::nn::ReLU()
};

torch::nn::Sequential ConvNetImpl::layer02{
    torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).bias(false)),
    torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.2)),
    torch::nn::BatchNorm2d(32),
    torch::nn::ReLU()
};

torch::nn::Sequential ConvNetImpl::layer03{
    torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).bias(false)),
    torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.2)),
    torch::nn::BatchNorm2d(64),
    torch::nn::ReLU(),
    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
};

torch::nn::Sequential ConvNetImpl::layer04{
    torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).bias(false)),
    torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.2)),
    torch::nn::BatchNorm2d(128),
    torch::nn::ReLU(),
    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
};

torch::nn::Sequential ConvNetImpl::layer05{
    torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).bias(false)),
    torch::nn::Dropout2d(torch::nn::Dropout2dOptions(0.2)),
    torch::nn::BatchNorm2d(256),
    torch::nn::ReLU()
};

torch::nn::AdaptiveAvgPool2d ConvNetImpl::pool{torch::nn::AdaptiveAvgPool2dOptions({4, 4})};

torch::nn::Sequential ConvNetImpl::layer06{
    torch::nn::Flatten(),
    torch::nn::Linear(256 * 4 * 4, 128),
    torch::nn::Dropout(torch::nn::DropoutOptions(0.2)),
    torch::nn::Sigmoid()
};


ConvNetImpl::ConvNetImpl(int64_t num_classes)
    : fc(128, num_classes) {
    register_module("layer010", layer01);
    register_module("layer020", layer02);
    register_module("layer030", layer03);
    register_module("layer040", layer04);
    register_module("layer050", layer05);
    register_module("pool", pool),
    register_module("layer060", layer06);
    register_module("fc", fc);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
    x = layer01->forward(x);
    x = layer02->forward(x);
    x = layer03->forward(x);
    x = layer04->forward(x);
    x = layer05->forward(x);
    x = pool->forward(x);
    x = x.view({-1,  256 * 4 * 4});
    x = layer06->forward(x);
    return fc->forward(x);
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
