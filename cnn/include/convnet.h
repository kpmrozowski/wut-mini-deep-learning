// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/nn/modules/padding.h>
#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
 public:
    explicit ConvNetImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer01{
        torch::nn::ReplicationPad2d(1),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 3).stride(1)),
        torch::nn::BatchNorm2d(8),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer02{
        torch::nn::ReplicationPad2d(1),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer03{
        torch::nn::ReplicationPad2d(1),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 3).stride(1)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer04{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer05{ // x2
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer06{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer07{ // x3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer08{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer09{ // x3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer10{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::ReLU(),
    };

    torch::nn::Sequential layer11{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 64, 3).stride(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
    };

    torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({4, 4})};

    torch::nn::Linear fc;
};

TORCH_MODULE(ConvNet);

// ########################################################################
// ################################ RUN 1 #################################
// ########################################################################

   //  torch::nn::Sequential layer1{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
   //      torch::nn::BatchNorm2d(16),
   //      torch::nn::ReLU(),
   //      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
   //  };

   //  torch::nn::Sequential layer2{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
   //      torch::nn::BatchNorm2d(32),
   //      torch::nn::ReLU(),
   //      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
   //  };

   //  torch::nn::Sequential layer3{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
   //      torch::nn::BatchNorm2d(64),
   //      torch::nn::ReLU(),
   //      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
   //  };

   //  torch::nn::Sequential layer4{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
   //      torch::nn::BatchNorm2d(128),
   //      torch::nn::ReLU(),
   //      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
   //  };

   //  torch::nn::Sequential layer5{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
   //      torch::nn::BatchNorm2d(256),
   //      torch::nn::ReLU(),
   //  };

// run:
// Training...
// Epoch [1/10], Trainset - Loss: 1.3394, Accuracy: 0.5414
// Epoch [2/10], Trainset - Loss: 0.9603, Accuracy: 0.6707
// Epoch [3/10], Trainset - Loss: 0.8511, Accuracy: 0.7068
// Epoch [4/10], Trainset - Loss: 0.7972, Accuracy: 0.7276
// Epoch [5/10], Trainset - Loss: 0.7616, Accuracy: 0.7403
// Epoch [6/10], Trainset - Loss: 0.7349, Accuracy: 0.7469
// Epoch [7/10], Trainset - Loss: 0.7183, Accuracy: 0.7538
// Epoch [8/10], Trainset - Loss: 0.7049, Accuracy: 0.7571
// Epoch [9/10], Trainset - Loss: 0.6860, Accuracy: 0.7626
// Epoch [10/10], Trainset - Loss: 0.6798, Accuracy: 0.7687
// Training finished!

// Testing...
// Testing finished!
// Testset - Loss: 0.6858, Accuracy: 0.7670



