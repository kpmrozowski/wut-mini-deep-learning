#include <torch/optim/schedulers/lr_scheduler.h>
#include <torch/optim/schedulers/step_lr.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <tuple>
#include "convnet.h"
#include "imagefolder_dataset.h"
#include <Util/CSVLogger.h>
#include <Util/Concurrency.h>
#include "imagefolder_dataset.h"
#ifdef MEASURE_TIME
#include <Util/Time.h>
#endif
#ifdef WITH_CUDA
#include <Eden_resources/Ngpus_Ncpus.h>
#endif
#define RUNS_COUNT 7

class MyScheluder : public torch::optim::LRScheduler {
public:

    MyScheluder(torch::optim::Optimizer& optimizer,
        const double base_lr)
    : torch::optim::LRScheduler(optimizer)
    , base_lr_(base_lr)
    , current_lr_(base_lr) {}

    double current_lr() { return current_lr_; }
private:
    std::vector<double> get_lrs() override {
        ++epoch;
        current_lr_ = 0.36 * (std::atan(-(epoch - 10) / 4) + M_PI_2) * base_lr_;
        auto new_lrs = this->get_current_lrs();
        std::fill(new_lrs.begin(), new_lrs.end(), current_lr_);
        return new_lrs;
    }
  const double base_lr_;
  uint epoch = 0;
  double current_lr_;

};

void client_threads::client_work(int run_idx)
{
#ifdef MEASURE_TIME
auto training_time = util::unix_time();
#endif
    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 3036;//2048;
    const size_t num_epochs = 20;
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;
    const uint64_t seed_cuda = 123;

    auto regularization_type = std::get<0>(setting);    // regularization::regularization_type
    auto regularization_lambda = std::get<1>(setting);  // double 
    auto augumentation_type = std::get<2>(setting);     // augumentation::augumentation_type
    auto experiment_name = std::get<3>(setting);        // string
    auto experiment_type_idx = std::get<4>(setting);    // int

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
#ifdef WITH_CUDA
    unsigned num_gpus = Eden_resources::get_gpus_count();
    unsigned gpu_idx = run_idx;
    if (cuda_available) device.set_index(gpu_idx < num_gpus ? gpu_idx : num_gpus - 1);
#endif
    torch::cuda::manual_seed(seed_cuda + run_idx);
    torch::cuda::manual_seed_all(seed_cuda + run_idx);  
    
    std::string experiment_run_name = "";
    experiment_run_name += "EXP_";
    experiment_run_name += std::to_string(experiment_type_idx);
    experiment_run_name += "_";
    experiment_run_name += experiment_name;
    experiment_run_name += "RUN_";
    experiment_run_name += std::to_string(run_idx);
    const std::string logs_path{std::string{"logs/"} + experiment_run_name};
    const std::string models_path{std::string{"models/"} + experiment_run_name};
    
    // Imagenette dataset
    auto train_dataset = augumentation::augumented_dataset(
        dataset::ImageFolderDataset(
            imagenette_data_path, 
            dataset::ImageFolderDataset::Mode::TRAIN,
            {160, 160}),
            augumentation_type)
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();
    
    auto test_dataset = 
        dataset::ImageFolderDataset(
            imagenette_data_path,
            dataset::ImageFolderDataset::Mode::VAL,
            {160, 160})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());
    
    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();
    
    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);
    
    double best_accuracy = 0;
    
    util::CSVLogger g_logger_training(logs_path + ".csv", 
        (std::string{"Epoch_of_"} + std::to_string(num_epochs)).c_str(),
        "train_loss", "train_accuracy", 
        "test_loss", "test_accuracy");
    
    // Model
    ConvNet model(num_classes);
    model->to(device);
    
    // Optimizer
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
    
    // Learning rate sheluder
    // torch::optim::StepLR scheduler{optimizer, 1, 0.1};
    MyScheluder scheduler{optimizer, learning_rate};
    
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "Training....\n";

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

            // Regularize
            if (regularization_type != regularization::regularization_type::none) {
                for (const auto &param : model->parameters()) {
                    loss += regularization_lambda
                        * (regularization_type == regularization::regularization_type::l1
                                                ? param.abs()
                                                : param.pow(2))
                            .sum();
                }
            }

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        } // batch loop
        scheduler.step();

        auto train_mean_loss = running_loss / num_train_samples;
        auto train_accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << train_mean_loss << ", train_accuracy: " << train_accuracy << ", lr: " << scheduler.current_lr();

        // <> Test the model
        model->eval();
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
        // </> Test the model
        
        double test_mean_loss = running_loss / num_test_samples;
        double test_accuracy = static_cast<double>(num_correct) / num_test_samples;
        
        std::cout << ", Testset - Loss: " << test_mean_loss << ", train_accuracy: " << test_accuracy << '\n';
        g_logger_training.log(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy);
        
        if (test_accuracy > best_accuracy) {
            best_accuracy = test_accuracy;
            std::cout << "Epoch " << epoch << " is the best so far! Saving to file...\n";
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(models_path);
        }
    } // epoch loop
    
    //<> Load best model for evaluation

    torch::serialize::InputArchive archive;
    archive.load_from(models_path);
    model->load(archive);
    
    std::cout << "Best model initial test...\n";
    
    model->eval();
    torch::InferenceMode no_grad;
    
    double running_loss = 0.0;
    uint64_t num_correct = 0;
    
    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }
    
    auto test_mean_loss = running_loss / num_test_samples;
    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    
    //<\> Load best model for evaluation
    
    std::cout << "Best Loss: " << test_mean_loss << ", Best_train_accuracy: " << test_accuracy << '\n';
    util::CSVLogger g_logger_testing((logs_path + "_test") + ".csv", "test_loss", "test_accuracy");
    g_logger_testing.log(test_mean_loss, test_accuracy);
    
#ifdef MEASURE_TIME
    fmt::print("training_time={}ms\n", util::unix_time() - training_time);
#endif
}

std::vector<SimulationSetting> prepare_settings() {
    regularization::regularization_type  regularization_type;
    double regularization_lambda;
    augumentation::augumentation_type augumentation_type;
    std::vector<SimulationSetting> settings;

    int experiment_type_idx = 6;
    std::vector<std::string> experiment_name(6, "");
    for (int reg_type_idx = 0; reg_type_idx < 3; ++reg_type_idx) {
        experiment_name.at(0) = "REG_";
        switch (reg_type_idx) {
        case 0:
            regularization_type = regularization::regularization_type::none;
            experiment_name.at(1) = "none_";
            break;
        case 1:
            regularization_type = regularization::regularization_type::l1;
            experiment_name.at(1) = "l1_";
            break;
        case 2:
            regularization_type = regularization::regularization_type::l2;
            experiment_name.at(1) = "l2_";
            break;
        }
        for (int reg_lambda_idx = 0; reg_lambda_idx < 2; ++reg_lambda_idx) {
            experiment_name.at(2) = "REGLAM_";
            if (regularization_type == regularization::regularization_type::none) {
                experiment_name.at(3) = "none";
                continue;
            }
            switch (reg_lambda_idx) {
            case 0:
                regularization_lambda = 1e-4;
                experiment_name.at(3) = "1e-4_";
                break;
            case 1:
                regularization_lambda = 1e-1;
                experiment_name.at(3) = "1e-1_";
                break;
            }
            for (int aug_type_idx = 0; aug_type_idx < 5; ++aug_type_idx) {
                experiment_name.at(4) = "AUG_";
                switch (aug_type_idx) {
                case 0:
                    augumentation_type = augumentation::augumentation_type::none;
                    experiment_name.at(5) = "none_";
                    break;
                case 1:
                    augumentation_type = augumentation::augumentation_type::flips;
                    experiment_name.at(5) = "flips_";
                    break;
                case 2:
                    augumentation_type = augumentation::augumentation_type::rotations;
                    experiment_name.at(5) = "rotations_";
                    break;
                case 3:
                    augumentation_type = augumentation::augumentation_type::crops;
                    experiment_name.at(5) = "crops_";
                    break;
                case 4:
                    augumentation_type = augumentation::augumentation_type::mixup;
                    experiment_name.at(5) = "mixup_";
                    break;
                }
                std::string exp_name = "";
                for (const auto& name_part : experiment_name) {
                    exp_name += name_part;
                }
                settings.emplace_back(
                    regularization_type,
                    regularization_lambda,
                    augumentation_type,
                    exp_name,
                    experiment_type_idx);
                ++experiment_type_idx;
            }
        }
    }
    return settings;
}

int main(int argc, char **argv) {
    std::cout << "Convolutional Neural Network\n\n";
    if (argc > 1) {
        std::string command{argv[1]};
        if ((command == "-h") | (command == "--help")) {
            fmt::print("HELP:\n./cnn <path_to_cifar-10>\n\n");
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
        client_threads parallel_runs{num_gpus, setting, argc > 1 ? argv[1] : "../../../cifar-10/"};
        fmt::print("311");
        parallel_runs.join_clients();
    }

    std::cout << "Training finished!\n\n";
}
