// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/padding.h>
#include <torch/nn/options/dropout.h>
#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
 public:
    explicit ConvNetImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer01{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
        torch::nn::Dropout2d(0.2),
        torch::nn::BatchNorm2d(16),
        torch::nn::Sigmoid(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer02{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
        torch::nn::Dropout2d(0.2),
        torch::nn::BatchNorm2d(32),
        torch::nn::Sigmoid(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer03{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
        torch::nn::Dropout2d(0.2),
        torch::nn::BatchNorm2d(64),
        torch::nn::Sigmoid(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer04{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
        torch::nn::Dropout2d(0.2),
        torch::nn::BatchNorm2d(128),
        torch::nn::Sigmoid(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer05{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
        torch::nn::BatchNorm2d(256),
        torch::nn::Sigmoid()
    };

   //  torch::nn::Sequential layer06{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(64),
   //      torch::nn::ReLU(),
   //  };

   //  torch::nn::Sequential layer07{ // x3
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(64),
   //      torch::nn::ReLU(),
   //  };

   //  torch::nn::Sequential layer08{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(128),
   //      torch::nn::ReLU(),
   //  };

   //  torch::nn::Sequential layer09{ // x3
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(128),
   //      torch::nn::ReLU(),
   //  };

   //  torch::nn::Sequential layer10{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(256),
   //      torch::nn::ReLU(),
   //  };

   //  torch::nn::Sequential layer11{
   //      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 64, 3).stride(1)),
   //      torch::nn::Dropout2d(0.2),
   //      torch::nn::BatchNorm2d(64),
   //      torch::nn::ReLU(),
   //  };

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


// ########################################################################
// ################################ RUN 2 #################################
// ########################################################################

//     torch::nn::Sequential layer01{
//         torch::nn::ReplicationPad2d(1),
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 3).stride(1)),
//         torch::nn::BatchNorm2d(8),
//         torch::nn::ReLU(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer02{
//         torch::nn::ReplicationPad2d(1),
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).stride(1)),
//         torch::nn::BatchNorm2d(16),
//         torch::nn::ReLU(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer03{
//         torch::nn::ReplicationPad2d(1),
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, 3).stride(1)),
//         torch::nn::BatchNorm2d(16),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer04{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
//         torch::nn::BatchNorm2d(32),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer05{ // x2
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1)),
//         torch::nn::BatchNorm2d(32),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer06{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
//         torch::nn::BatchNorm2d(64),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer07{ // x3
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
//         torch::nn::BatchNorm2d(64),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer08{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
//         torch::nn::BatchNorm2d(128),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer09{ // x3
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
//         torch::nn::BatchNorm2d(128),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer10{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
//         torch::nn::BatchNorm2d(256),
//         torch::nn::ReLU(),
//     };

//     torch::nn::Sequential layer11{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 64, 3).stride(1)),
//         torch::nn::BatchNorm2d(64),
//         torch::nn::ReLU(),
//     };

//     torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({4, 4})};
// Training...
// Epoch [1/10], Trainset - Loss: 1.8170, Accuracy: 0.3245
// Epoch [2/10], Trainset - Loss: 1.5711, Accuracy: 0.4270
// Epoch [3/10], Trainset - Loss: 1.3787, Accuracy: 0.5020
// Epoch [4/10], Trainset - Loss: 1.2461, Accuracy: 0.5519
// Epoch [5/10], Trainset - Loss: 1.1531, Accuracy: 0.5874
// Epoch [6/10], Trainset - Loss: 1.0945, Accuracy: 0.6083
// Epoch [7/10], Trainset - Loss: 1.0528, Accuracy: 0.6264
// Epoch [8/10], Trainset - Loss: 1.0224, Accuracy: 0.6368
// Epoch [9/10], Trainset - Loss: 0.9915, Accuracy: 0.6482
// Epoch [10/10], Trainset - Loss: 0.9680, Accuracy: 0.6576
// Training finished!

// Testing...
// Testing finished!
// Testset - Loss: 6.8402, Accuracy: 0.2168


// ########################################################################
// ################################ RUN 3 #################################
// ########################################################################


//     torch::nn::Sequential layer01{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)),
//         torch::nn::Dropout2d(0.2),
//         torch::nn::BatchNorm2d(16),
//         torch::nn::Sigmoid(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer02{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)),
//         torch::nn::Dropout2d(0.2),
//         torch::nn::BatchNorm2d(32),
//         torch::nn::Sigmoid(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer03{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1)),
//         torch::nn::Dropout2d(0.2),
//         torch::nn::BatchNorm2d(64),
//         torch::nn::Sigmoid(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer04{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
//         torch::nn::Dropout2d(0.2),
//         torch::nn::BatchNorm2d(128),
//         torch::nn::Sigmoid(),
//         torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//     };

//     torch::nn::Sequential layer05{
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
//         torch::nn::BatchNorm2d(256),
//         torch::nn::Sigmoid()
//     };

// Training...
// Epoch [1/30], Trainset - Loss: 2.1059, Accuracy: 0.2731
// Epoch [2/30], Trainset - Loss: 1.9064, Accuracy: 0.3348
// Epoch [3/30], Trainset - Loss: 1.8043, Accuracy: 0.3694
// Epoch [4/30], Trainset - Loss: 1.7366, Accuracy: 0.3958
// Epoch [5/30], Trainset - Loss: 1.6589, Accuracy: 0.4226
// Epoch [6/30], Trainset - Loss: 1.6005, Accuracy: 0.4481
// Epoch [7/30], Trainset - Loss: 1.5281, Accuracy: 0.4702
// Epoch [8/30], Trainset - Loss: 1.4979, Accuracy: 0.4821
// Epoch [9/30], Trainset - Loss: 1.4692, Accuracy: 0.4938
// Epoch [10/30], Trainset - Loss: 1.4462, Accuracy: 0.5017
// Epoch [11/30], Trainset - Loss: 1.4240, Accuracy: 0.5088
// Epoch [12/30], Trainset - Loss: 1.4187, Accuracy: 0.5118
// Epoch [13/30], Trainset - Loss: 1.4039, Accuracy: 0.5198
// Epoch [14/30], Trainset - Loss: 1.3800, Accuracy: 0.5253
// Epoch [15/30], Trainset - Loss: 1.3741, Accuracy: 0.5282
// Epoch [16/30], Trainset - Loss: 1.3626, Accuracy: 0.5311
// Epoch [17/30], Trainset - Loss: 1.3474, Accuracy: 0.5340
// Epoch [18/30], Trainset - Loss: 1.3459, Accuracy: 0.5397
// Epoch [19/30], Trainset - Loss: 1.3463, Accuracy: 0.5399
// Epoch [20/30], Trainset - Loss: 1.3381, Accuracy: 0.5414
// Epoch [21/30], Trainset - Loss: 1.3323, Accuracy: 0.5437
// Epoch [22/30], Trainset - Loss: 1.3292, Accuracy: 0.5429
// Epoch [23/30], Trainset - Loss: 1.3245, Accuracy: 0.5474
// Epoch [24/30], Trainset - Loss: 1.3147, Accuracy: 0.5481
// Epoch [25/30], Trainset - Loss: 1.3115, Accuracy: 0.5493
// Epoch [26/30], Trainset - Loss: 1.3162, Accuracy: 0.5489
// Epoch [27/30], Trainset - Loss: 1.3059, Accuracy: 0.5510
// Epoch [28/30], Trainset - Loss: 1.3003, Accuracy: 0.5530
// Epoch [29/30], Trainset - Loss: 1.3097, Accuracy: 0.5526
// Epoch [30/30], Trainset - Loss: 1.2949, Accuracy: 0.5550
// Training finished!

// Testing...
// Testing finished!
// Testset - Loss: 1.7731, Accuracy: 0.4520