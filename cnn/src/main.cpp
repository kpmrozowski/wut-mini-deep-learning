// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "convnet.h"
#include "imagefolder_dataset.h"

using dataset::ImageFolderDataset;

int main(int argc, char **argv) {
    std::cout << "Convolutional Neural Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 16;
    const size_t num_epochs = 30;
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;
    const uint64_t seed_cuda = 123;
    
    torch::cuda::manual_seed(seed_cuda);
    torch::cuda::manual_seed_all(seed_cuda);
    const std::string imagenette_data_path = argc >= 2 ? argv[1] : "../../../../cifar-10";

    // Imagenette dataset
    auto train_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::TRAIN, {160, 160})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = ImageFolderDataset(imagenette_data_path, ImageFolderDataset::Mode::VAL, {160, 160})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Model
    ConvNet model(num_classes);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    size_t best_correct = 0;

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Train the model
        model->train();

        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy;

        // Test the model
        model->eval();
        torch::InferenceMode no_grad;

        running_loss = 0.0;
        num_correct = 0;

        for (const auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto output = model->forward(data);

            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.item<double>() * data.size(0);

            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();
        }

        auto test_sample_mean_loss = running_loss / num_test_samples;
        auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;

        std::cout << ", Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

        if (num_correct > best_correct) {
            best_correct = num_correct;
            std::cout << "Best epoch so far! Saving to file...\n";
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to("best_model");
        }
    }

    std::cout << "Training finished!\n\n";
}
