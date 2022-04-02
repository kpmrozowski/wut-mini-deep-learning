// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/dropout.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/padding.h>
#include <torch/nn/options/dropout.h>
#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
 public:
    explicit ConvNetImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

    void print_modules();

 private:
    static torch::nn::Sequential layer01;
    static torch::nn::Sequential layer02;
    static torch::nn::Sequential layer03;
    static torch::nn::Sequential layer04;
    static torch::nn::Sequential layer05;
    static torch::nn::AdaptiveAvgPool2d pool;
    static torch::nn::Sequential layer06;
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
// ############ VERSION v1.0.0 RUN 1 accuracy 55% #########################
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


// ########################################################################
// ############ VERSION v1.0.0 RUN 2 accuracy 61% #########################
// ########################################################################

// CUDA available. Training on GPU.
// Training....
// Epoch [1/1000], Trainset - Loss: 1.9484, Accuracy: 0.3036, Testset - Loss: 1.7661, Accuracy: 0.3764
// Best epoch so far! Saving to file...
// Epoch [2/1000], Trainset - Loss: 1.7720, Accuracy: 0.3676, Testset - Loss: 1.5826, Accuracy: 0.4240
// Best epoch so far! Saving to file...
// Epoch [3/1000], Trainset - Loss: 1.6598, Accuracy: 0.4065, Testset - Loss: 1.5585, Accuracy: 0.4458
// Best epoch so far! Saving to file...
// Epoch [4/1000], Trainset - Loss: 1.5424, Accuracy: 0.4517, Testset - Loss: 1.6371, Accuracy: 0.4062
// Epoch [5/1000], Trainset - Loss: 1.4508, Accuracy: 0.4896, Testset - Loss: 1.6057, Accuracy: 0.4298
// Epoch [6/1000], Trainset - Loss: 1.4070, Accuracy: 0.5064, Testset - Loss: 1.4506, Accuracy: 0.4726
// Best epoch so far! Saving to file...
// Epoch [7/1000], Trainset - Loss: 1.3842, Accuracy: 0.5150, Testset - Loss: 1.6897, Accuracy: 0.4186
// Epoch [8/1000], Trainset - Loss: 1.3521, Accuracy: 0.5293, Testset - Loss: 1.5594, Accuracy: 0.4474
// Epoch [9/1000], Trainset - Loss: 1.3341, Accuracy: 0.5334, Testset - Loss: 1.6216, Accuracy: 0.4630
// Epoch [10/1000], Trainset - Loss: 1.3179, Accuracy: 0.5389, Testset - Loss: 1.8216, Accuracy: 0.3816
// Epoch [11/1000], Trainset - Loss: 1.3090, Accuracy: 0.5420, Testset - Loss: 1.1936, Accuracy: 0.5798
// Best epoch so far! Saving to file...
// Epoch [12/1000], Trainset - Loss: 1.2924, Accuracy: 0.5487, Testset - Loss: 2.9033, Accuracy: 0.3114
// Epoch [13/1000], Trainset - Loss: 1.2828, Accuracy: 0.5509, Testset - Loss: 1.4341, Accuracy: 0.5238
// Epoch [14/1000], Trainset - Loss: 1.2713, Accuracy: 0.5539, Testset - Loss: 1.9378, Accuracy: 0.3994
// Epoch [15/1000], Trainset - Loss: 1.2729, Accuracy: 0.5557, Testset - Loss: 1.5909, Accuracy: 0.4954
// Epoch [16/1000], Trainset - Loss: 1.2593, Accuracy: 0.5597, Testset - Loss: 1.2222, Accuracy: 0.5648
// Epoch [17/1000], Trainset - Loss: 1.2653, Accuracy: 0.5592, Testset - Loss: 1.3044, Accuracy: 0.5542
// Epoch [18/1000], Trainset - Loss: 1.2476, Accuracy: 0.5621, Testset - Loss: 1.4633, Accuracy: 0.5108
// Epoch [19/1000], Trainset - Loss: 1.2503, Accuracy: 0.5647, Testset - Loss: 1.3246, Accuracy: 0.5324
// Epoch [20/1000], Trainset - Loss: 1.2435, Accuracy: 0.5660, Testset - Loss: 1.3351, Accuracy: 0.5344
// Epoch [21/1000], Trainset - Loss: 1.2254, Accuracy: 0.5736, Testset - Loss: 1.4007, Accuracy: 0.5184
// Epoch [22/1000], Trainset - Loss: 1.2335, Accuracy: 0.5680, Testset - Loss: 2.3623, Accuracy: 0.3252
// Epoch [23/1000], Trainset - Loss: 1.2253, Accuracy: 0.5731, Testset - Loss: 1.3501, Accuracy: 0.5310
// Epoch [24/1000], Trainset - Loss: 1.2119, Accuracy: 0.5766, Testset - Loss: 1.4339, Accuracy: 0.4966
// Epoch [25/1000], Trainset - Loss: 1.2064, Accuracy: 0.5816, Testset - Loss: 1.9057, Accuracy: 0.4402
// Epoch [26/1000], Trainset - Loss: 1.2071, Accuracy: 0.5804, Testset - Loss: 1.9590, Accuracy: 0.4182
// Epoch [27/1000], Trainset - Loss: 1.2135, Accuracy: 0.5782, Testset - Loss: 1.4504, Accuracy: 0.5248
// Epoch [28/1000], Trainset - Loss: 1.2001, Accuracy: 0.5799, Testset - Loss: 1.5078, Accuracy: 0.5308
// Epoch [29/1000], Trainset - Loss: 1.1982, Accuracy: 0.5823, Testset - Loss: 1.5073, Accuracy: 0.5452
// Epoch [30/1000], Trainset - Loss: 1.1786, Accuracy: 0.5895, Testset - Loss: 1.3825, Accuracy: 0.5054
// Epoch [31/1000], Trainset - Loss: 1.1960, Accuracy: 0.5830, Testset - Loss: 1.1640, Accuracy: 0.6058
// Best epoch so far! Saving to file...
// Epoch [32/1000], Trainset - Loss: 1.1858, Accuracy: 0.5888, Testset - Loss: 1.4652, Accuracy: 0.4968
// Epoch [33/1000], Trainset - Loss: 1.1864, Accuracy: 0.5873, Testset - Loss: 1.2306, Accuracy: 0.5816
// Epoch [34/1000], Trainset - Loss: 1.1658, Accuracy: 0.5927, Testset - Loss: 3.2710, Accuracy: 0.2566
// Epoch [35/1000], Trainset - Loss: 1.1658, Accuracy: 0.5925, Testset - Loss: 2.1480, Accuracy: 0.3970
// Epoch [36/1000], Trainset - Loss: 1.1622, Accuracy: 0.5964, Testset - Loss: 1.7896, Accuracy: 0.4714
// Epoch [37/1000], Trainset - Loss: 1.1553, Accuracy: 0.5959, Testset - Loss: 1.9960, Accuracy: 0.4154
// Epoch [38/1000], Trainset - Loss: 1.1519, Accuracy: 0.6021, Testset - Loss: 1.4062, Accuracy: 0.4976
// Epoch [39/1000], Trainset - Loss: 1.1729, Accuracy: 0.5931, Testset - Loss: 2.1496, Accuracy: 0.3444
// Epoch [40/1000], Trainset - Loss: 1.1500, Accuracy: 0.6008, Testset - Loss: 1.4632, Accuracy: 0.5020
// Epoch [41/1000], Trainset - Loss: 1.1492, Accuracy: 0.6018, Testset - Loss: 1.6171, Accuracy: 0.4922
// Epoch [42/1000], Trainset - Loss: 1.1476, Accuracy: 0.5988, Testset - Loss: 1.6462, Accuracy: 0.4426
// Epoch [43/1000], Trainset - Loss: 1.1427, Accuracy: 0.6015, Testset - Loss: 1.7843, Accuracy: 0.4594
// Epoch [44/1000], Trainset - Loss: 1.1367, Accuracy: 0.6051, Testset - Loss: 1.3360, Accuracy: 0.5364
// Epoch [45/1000], Trainset - Loss: 1.1433, Accuracy: 0.6004, Testset - Loss: 2.9395, Accuracy: 0.3404
// Epoch [46/1000], Trainset - Loss: 1.1349, Accuracy: 0.6047, Testset - Loss: 3.1447, Accuracy: 0.2622
// Epoch [47/1000], Trainset - Loss: 1.1380, Accuracy: 0.6055, Testset - Loss: 1.8720, Accuracy: 0.3940
// Epoch [48/1000], Trainset - Loss: 1.1220, Accuracy: 0.6105, Testset - Loss: 1.8749, Accuracy: 0.3860
// Epoch [49/1000], Trainset - Loss: 1.1172, Accuracy: 0.6116, Testset - Loss: 1.2108, Accuracy: 0.5908
// Epoch [50/1000], Trainset - Loss: 1.1267, Accuracy: 0.6078, Testset - Loss: 2.5397, Accuracy: 0.3502