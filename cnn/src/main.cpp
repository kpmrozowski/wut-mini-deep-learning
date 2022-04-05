#include "convnet.h"
#include <ATen/core/interned_strings.h>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <train_options.h>
#include <Util/Concurrency.h>
#include <imagefolder_dataset.h>
#include <Util/CSVLogger.h>
#include <Util/CSVReader.h>
#include <iostream>
#ifdef ON_EDEN
#include <Eden_resources/Ngpus_Ncpus.h>
#endif

const uint64_t img_width = 32;
const uint64_t img_height = 32;
#ifndef ON_EDEN
const int64_t batch_size = 256;
#else
const int64_t batch_size = 4048;//=39GB;
#endif
const int64_t data_workers = 1;

using Mode = dataset::ImageFolderDataset::Mode;
using Ensemble = std::map<int, std::array<std::pair<double, ConvNet>, 3>>;
using Dataset = torch::data::datasets::MapDataset<
    torch::data::datasets::MapDataset<
        dataset::ImageFolderDataset,
        torch::data::transforms::Normalize<>
    >, torch::data::transforms::Stack<>
>;
using Loader = std::unique_ptr<
    torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            torch::data::datasets::MapDataset<
                dataset::ImageFolderDataset,
                torch::data::transforms::Normalize<>
            >,
            torch::data::transforms::Stack<>
        >,
        torch::data::samplers::SequentialSampler
    >
>;

inline std::pair<double, double> evaluate(Ensemble& models, const Dataset& dataset, const torch::Device& device) {
    const int64_t bs = 1;
    auto num_samples = dataset.size().value();
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset), torch::data::DataLoaderOptions(bs));

    torch::InferenceMode no_grad;
    uint64_t num_correct = 0;
    double running_loss = 0.0;
    
    for (const auto& batch : *loader) {
        auto data = batch.data.to(torch::Device(torch::kCPU));
        auto target = batch.target.to(torch::Device(torch::kCPU));

        // std::array<double, 10> score = {};
        torch::Tensor scores = torch::zeros(10);

        for (auto &model_data : models) {
            for (int i = 0; i < 3; ++i) {
                auto output = model_data.second[i].second->forward(data);
                auto prediction = output.argmax(1);
                std::cout << "prediction: " << prediction << std::endl;
                if (prediction.item<int>() == model_data.first) {
                    scores[model_data.first] += model_data.second[i].first; // Here use the model quality
                }
            }
        }
        // int ensemble_prediction = std::distance(score.begin(), std::max_element(score.begin(), score.end()));
        auto loss = torch::nn::functional::cross_entropy(scores, target);
        running_loss += loss.template item<double>() * data.size(0);
        auto prediction = scores.argmax(1);
        num_correct += prediction.eq(target).sum().template item<int64_t>();
    }
    
    auto accuracy = static_cast<double>(num_correct) / num_samples;
    return std::make_pair(running_loss, accuracy);
}

inline void ensemble(int argc, char **argv) {
    std::string cifar_path{argc > 2 ? argv[2] : CIFAR_PATH};
    
    auto train_dataset = 
        dataset::ImageFolderDataset(
            cifar_path,
            Mode::TRAIN,
            {img_height, img_width})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());
    
    auto val_dataset = 
        dataset::ImageFolderDataset(
            cifar_path,
            Mode::VAL,
            {img_height, img_width})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());
    
    // auto test_dataset =
    //     dataset::ImageFolderDataset(
    //         cifar_path,
    //         Mode::TEST,
    //         {img_height, img_width})
    //     .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
    //     .map(torch::data::transforms::Stack<>());
    // auto num_test_samples = test_dataset.size().value();
    // auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    //     std::move(test_dataset), torch::data::DataLoaderOptions(batch_size).workers(data_workers));

    // Device
#ifdef ON_CUDA
    bool cuda_available = torch::cuda::is_available();
#else
    bool cuda_available = false;
#endif
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
#ifdef ON_EDEN
    unsigned num_gpus = Eden_resources::get_gpus_count();
    unsigned gpu_idx = 0; // ???
    if (cuda_available) device.set_index(gpu_idx < num_gpus ? gpu_idx : num_gpus - 1);
#endif

    std::map<std::string, int> label_to_idx{
        {"airplane", 0},
        {"automobile", 1},
        {"bird", 2},
        {"cat", 3},
        {"deer", 4},
        {"dog", 5},
        {"frog", 6},
        {"horse", 7},
        {"ship", 8},
        {"truck", 9},
    };
    Ensemble models;
    for (auto label_idx : label_to_idx) {
        auto label = label_idx.first;
        torch::serialize::InputArchive archive;
        
        double model1_acc = util::read_record(
            std::string(LOGS_PATH) + "EXP_26_REG_none_REGLAM_none_AUG_none_" + label + "_test.csv", 1, 1);
        archive.load_from(std::string(MODELS_PATH) + "EXP_26_REG_none_REGLAM_none_AUG_none_" + label);
        ConvNet model1;
        model1->load(archive);
        model1->eval();

        double model2_acc = util::read_record(
            std::string(LOGS_PATH) + "EXP_27_REG_none_REGLAM_none_AUG_crops_" + label + "_test.csv", 1, 1);
        archive.load_from(std::string(MODELS_PATH) + "EXP_27_REG_none_REGLAM_none_AUG_crops_" + label);
        ConvNet model2;
        model2->eval();

        double model3_acc = util::read_record(
            std::string(LOGS_PATH) + "EXP_28_REG_none_REGLAM_none_AUG_colors_" + label + "_test.csv", 1, 1);
        archive.load_from(std::string(MODELS_PATH) + "EXP_28_REG_none_REGLAM_none_AUG_colors_" + label);
        ConvNet model3;
        model3->eval();

        models.insert({label_idx.second, {
            std::make_pair(model1_acc, model1),
            std::make_pair(model2_acc, model2),
            std::make_pair(model3_acc, model3)}});
    }
    auto [train_loss, train_accuracy] = evaluate(models, train_dataset, device);
    auto [val_loss, val_accuracy] = evaluate(models, val_dataset, device);
    
    fmt::print("Ensemble_loss:\n\ttrain={:0.6f}\n\tval={:0.6f}\n", train_loss, val_loss);
    fmt::print("Ensemble_accuracy:\n\ttrain={:0.6f}\n\tval={:0.6f}\n", train_accuracy, val_accuracy);
    util::CSVLogger g_logger_validating(std::string(LOGS_PATH) + "ensemble_acc.csv", "train_loss", "train_accuracy", "val_loss", "val_accuracy");
    g_logger_validating.log(train_loss, train_accuracy, val_loss, val_accuracy);
}

void training(int argc, char **argv) {
    std::vector<SimulationSetting> settings = prepare_settings();
    fmt::print("settings size = {}\n", settings.size());
    for (const auto& setting : settings) {    
        // auto cuda_available = torch::cuda::is_available();
#ifdef ON_EDEN
        unsigned num_gpus = Eden_resources::get_gpus_count();
#else
        unsigned num_gpus = 1;
#endif
        std::string cifar_path{argc > 2 ? argv[2] : CIFAR_PATH};
        fmt::print("cifar path: {}\n", CIFAR_PATH);
        client_threads parallel_runs{num_gpus, setting, CIFAR_PATH};
        parallel_runs.join_clients();
    }

    std::cout << "Training finished!\n\n";
}

int main(int argc, char **argv) {
    std::cout << "Convolutional Neural Network\n\n";
    if (argc > 1) {
        std::string command{argv[1]};
        if ((command == "-h") | (command == "--help")) {
            fmt::print("HELP:\n./cnn <path_to_cifar-10>\n\n");
        }
        if ((command == "-t") | (command == "--training")) {
            training(argc, argv);
            return EXIT_SUCCESS;
        }
    }
    ensemble(argc, argv);
    return EXIT_SUCCESS;
}
