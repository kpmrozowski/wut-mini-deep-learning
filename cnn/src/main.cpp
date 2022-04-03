#include "convnet.h"
#include <ATen/core/interned_strings.h>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <train_options.h>
#include <Util/Concurrency.h>
#include <imagefolder_dataset.h>
#include <Util/CSVLogger.h>
#ifdef WITH_CUDA
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

void ensemble(int argc, char **argv) {
    std::string cifar_path{argc > 2 ? argv[2] : CIFAR_PATH};
    auto test_dataset =
        dataset::ImageFolderDataset(
            cifar_path,
            dataset::ImageFolderDataset::Mode::VAL,
            {img_height, img_width})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());
    auto num_test_samples = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions(batch_size).workers(data_workers));

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
#ifdef WITH_CUDA
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
    std::map<int, std::array<ConvNet, 3>> models;
    for (auto label_idx : label_to_idx) {
        auto label = label_idx.first;

        torch::serialize::InputArchive archive;
        archive.load_from(std::string(MODELS_PATH) + "EXP_26_REG_none_REGLAM_none_AUG_none_" + label);
        ConvNet model1;
        model1->load(archive);
        model1->eval();

        archive.load_from(std::string(MODELS_PATH) + "EXP_27_REG_none_REGLAM_none_AUG_crop_" + label);
        ConvNet model2;
        model2->eval();

        archive.load_from(std::string(MODELS_PATH) + "EXP_28_REG_none_REGLAM_none_AUG_colors_" + label);
        ConvNet model3;
        model3->eval();

        models.insert({label_idx.second, {model1, model2, model3}});
    }

    torch::InferenceMode no_grad;

    uint64_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target;

        std::array<float, 10> score = {};

        for (auto &model_data : models) {
            for (int i = 0; i < 3; ++i) {
                auto output = model_data.second[i]->forward(data);
                auto prediction = output.argmax(1);
                if (prediction.item<int>() == model_data.first) {
                    score[model_data.first] += 1; // TODO: Here use the model quality
                }
            }
        }

        auto final_prediction = std::max_element(score.begin(), score.end());
        num_correct += *final_prediction == target.item<float>();
    }

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;

    //<\> Load best model for evaluation

    fmt::print("Ensemble_test_accuracy: {:0.6f}\n", test_accuracy);
    util::CSVLogger g_logger_testing((std::string(LOGS_PATH) + "_ensemble_test") + ".csv", "test_accuracy");
    g_logger_testing.log(test_accuracy);

}

int main(int argc, char **argv) {
    std::cout << "Convolutional Neural Network\n\n";
    if (argc > 1) {
        std::string command{argv[1]};
        if ((command == "-h") | (command == "--help")) {
            fmt::print("HELP:\n./cnn <path_to_cifar-10>\n\n");
        }
        if ((command == "-e") | (command == "--ensemble")) {
            ensemble(argc, argv);
            return EXIT_SUCCESS;
        }
    }
    std::vector<SimulationSetting> settings = prepare_settings();
    fmt::print("settings size = {}\n", settings.size());
    for (const auto& setting : settings) {    
        // auto cuda_available = torch::cuda::is_available();
#ifdef WITH_CUDA
        unsigned num_gpus = Eden_resources::get_gpus_count();
#else
        unsigned num_gpus = 1;
#endif
        std::string label{argc > 1 ? argv[1] : ""};
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
        std::string cifar_path{argc > 2 ? argv[2] : CIFAR_PATH};
        cifar_path += "../ensemble-cifar/" + std::to_string(label_to_idx[label]);
        fmt::print("cifar path: {}\n", cifar_path);
        client_threads parallel_runs{num_gpus, setting, label, cifar_path};
        parallel_runs.join_clients();
    }

    std::cout << "Training finished!\n\n";
}
