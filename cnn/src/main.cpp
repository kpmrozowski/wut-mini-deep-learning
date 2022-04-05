#include "convnet.h"
#include <ATen/core/interned_strings.h>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <train_options.h>
// #include <Util/Concurrency.h>
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
const std::vector<std::string> idx_to_label{
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
};

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

inline std::tuple<double, std::vector<std::pair<uint64_t, std::string>>> 
evaluate(ConvNet& model, const Dataset& dataset, const torch::Device& device) {
    const int64_t bs = 1;
    auto num_samples = dataset.size().value();
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset), torch::data::DataLoaderOptions(bs));

    torch::InferenceMode no_grad;
    uint64_t num_correct = 0;
    // double running_loss = 0.0;
    std::vector<std::pair<uint64_t, std::string>> predicted_labels;
    predicted_labels.reserve(num_samples);
    uint64_t count = 0;
    
    for (const auto& batch : *loader) {
        torch::Tensor scores = torch::zeros(10);
        auto output = model->forward(batch.data);
        auto prediction = output.argmax(0);
        uint64_t pred_idx = prediction.template item<int64_t>();
        std::string pred_label = idx_to_label[pred_idx];
        predicted_labels.emplace_back(count, pred_label);
        fmt::print("{} ", ++count);
    }
    auto accuracy = static_cast<double>(num_correct) / num_samples;
    return std::make_tuple(accuracy, predicted_labels);
}

inline void ensemble(int argc, char **argv) {
    std::string cifar_path{argc > 2 ? argv[2] : CIFAR_PATH};
    auto test_dataset =
        dataset::ImageFolderDataset(
            cifar_path,
            Mode::TEST,
            {img_height, img_width})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());

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
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(MODELS_PATH) + "EXP_15_REG_none_REGLAM_none_AUG_crops_RUN_0");
    ConvNet model1;
    model1->load(archive);
    model1->to(device);
    model1->eval();
    auto [test_acc, preds_test] = evaluate(model1, test_dataset, device);
    
    util::CSVLogger g_logger_predictions_train(std::string(LOGS_PATH) + "crops_predictions_test.csv", "id", "class");
    for (const auto& pred : preds_test)
        g_logger_predictions_train.log(pred.first, pred.second);
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
        // client_threads parallel_runs{num_gpus, setting, CIFAR_PATH};
        // parallel_runs.join_clients();
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
