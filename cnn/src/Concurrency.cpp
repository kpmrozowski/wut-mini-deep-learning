#include <Util/Concurrency.h>
#include <torch/data/dataloader_options.h>
#include <torch/optim/schedulers/step_lr.h>
#include <torch/torch.h>
#include <convnet.h>
#include <imagefolder_dataset.h>
#include <Util/CSVLogger.h>
#ifdef MEASURE_TIME
#include <Util/Time.h>
#endif
#include <train_options.h>

void client_threads::client_work(int run_idx)
{
   at::init_num_threads();
#ifdef MEASURE_TIME
   auto training_time = util::unix_time();
#endif
    // Hyper parameters
    const int64_t num_classes = 10;
#ifndef ON_EDEN
    const int64_t batch_size = 256;
#else
    const int64_t batch_size = 4048;//=39GB;
#endif
    const int64_t data_workers = 1;
    const size_t num_epochs = 200;
    const double learning_rate = 4e-3;
    const double learning_rate_multiplayer = 0.83;
    const uint learning_rate_decay_step = 10;
    const double weight_decay = 1e-3;
    const uint64_t seed_cuda = 123;
    const uint64_t img_width = 32;
    const uint64_t img_height = 32;

    auto regularization_type = std::get<0>(setting);    // regularization::regularization_type
    auto regularization_lambda = std::get<1>(setting);  // double 
    auto augumentation_type = std::get<2>(setting);     // augumentation::augumentation_type
    auto experiment_name = std::get<3>(setting);        // string
    auto experiment_type_idx = std::get<4>(setting);    // int
    
    std::string experiment_run_name = "";
    experiment_run_name += "EXP_";
    experiment_run_name += std::to_string(experiment_type_idx);
    experiment_run_name += "_";
    experiment_run_name += experiment_name;
    experiment_run_name += "RUN_";
    experiment_run_name += std::to_string(run_idx);
    const std::string logs_path{std::string{LOGS_PATH} + experiment_run_name};
    const std::string models_path{std::string{MODELS_PATH} + experiment_run_name};

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
    
    // Imagenette dataset
    auto train_dataset = augumentation::augumented_dataset(
        dataset::ImageFolderDataset(
            imagenette_data_path, 
            dataset::ImageFolderDataset::Mode::TRAIN,
            {img_height, img_width}),
            augumentation_type)
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();
    
    auto test_dataset = 
        dataset::ImageFolderDataset(
            imagenette_data_path,
            dataset::ImageFolderDataset::Mode::VAL,
            {img_height, img_width})
        .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
        .map(torch::data::transforms::Stack<>());
    
    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();
    
    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions(batch_size).workers(data_workers));
    
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions(batch_size).workers(data_workers));
    
    double best_accuracy = 0;
    double current_lr = learning_rate;
    
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
    torch::optim::StepLR scheduler{optimizer, learning_rate_decay_step, learning_rate_multiplayer};
    // MyScheluder scheduler{optimizer, learning_rate};
    
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    fmt::print("Training....\n");

    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
#ifdef MEASURE_TIME
        int64_t time = util::unix_time();
#endif
        // Train the model
        model->train();
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
#ifdef MEASURE_TIME
            int64_t batch_time = util::unix_time();
#endif
            // Transfer images and target labels to device
            auto target = batch.target.to(device, true);
#ifdef MEASURE_TIME
            fmt::print("\ttarget_to_device_time: {} ms\n", util::unix_time() - batch_time);
#endif
            auto data = batch.data.to(device, true);
#ifdef MEASURE_TIME
            fmt::print("\tdata_to_device_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Forward pass
            auto output = model->forward(data);
#ifdef MEASURE_TIME
            fmt::print("\tforward_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);
#ifdef MEASURE_TIME
            fmt::print("\tentropy_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Update running loss
            running_loss += loss.item<double>() * batch.data.size(0);
#ifdef MEASURE_TIME
            fmt::print("\trunning_loss_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Calculate prediction
            auto prediction = output.argmax(1);
#ifdef MEASURE_TIME
            fmt::print("\targmax_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();
#ifdef MEASURE_TIME
            fmt::print("\tsum_time: {} ms\n", util::unix_time() - batch_time);
#endif
            // Regularize
            if (regularization_type != regularization::regularization_type::none) {
                for (const auto &param : model->parameters()) {
                    loss += regularization_lambda
                        * (regularization_type == regularization::regularization_type::l1
                                                ? param.abs()
                                                : param.pow(2))
                            .sum();
                }
#ifdef MEASURE_TIME
                fmt::print("\tregularization_time: {} ms\n", util::unix_time() - batch_time);
#endif
            }
            // Backward pass and optimize
            optimizer.zero_grad();
#ifdef MEASURE_TIME
            fmt::print("\tzero_grad_time: {} ms\n", util::unix_time() - batch_time);
#endif
            loss.backward();
#ifdef MEASURE_TIME
            fmt::print("\tbackward_time: {} ms\n", util::unix_time() - batch_time);
#endif
            optimizer.step();
#ifdef MEASURE_TIME
            fmt::print("\tstep_time: {} ms\n\n", util::unix_time() - batch_time);
#endif
        } // batch loop
        // scheduler.set_epoch(epoch);
        auto train_mean_loss = running_loss / num_train_samples;
        auto train_accuracy = static_cast<double>(num_correct) / num_train_samples;
#ifdef MEASURE_TIME
        fmt::print("batches_time: {} ms\n", util::unix_time() - time);
#endif
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
#ifdef MEASURE_TIME
        fmt::print("eval_time: {} ms\n", util::unix_time() - time);
#endif
        // </> Test the model
        
        double test_mean_loss = running_loss / num_test_samples;
        double test_accuracy = static_cast<double>(num_correct) / num_test_samples;
        
        fmt::print("Epoch [{}/{}], Trainset - Loss: {:0.4f}, train_accuracy: {:0.4f}, Testset - Loss: {:0.4f}, test_accuracy: {:0.4f}, lr: {:0.6f}\n",
            epoch + 1, num_epochs, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy, current_lr);
        g_logger_training.log(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy);
        
        if (test_accuracy > best_accuracy) {
            best_accuracy = test_accuracy;
            fmt::print("Epoch {} is the best so far! Saving to file...\n", (epoch+1));
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(models_path);
            fmt::print("Epoch {} saved!\n", (epoch+1));
        }
        scheduler.step();
        if ((epoch+1) % learning_rate_decay_step == 0) {
            current_lr *= learning_rate_multiplayer;
        }
#ifdef MEASURE_TIME
        fmt::print("saving_model_time: {} ms\n", util::unix_time() - time);
#endif
    } // epoch loop
    
    //<> Load best model for evaluation

    torch::serialize::InputArchive archive;
    archive.load_from(models_path);
    model->load(archive);
    fmt::print("Best model initial test...\n");
    
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
    
    fmt::print("Best Loss: {:0.6f}, Best_test_accuracy: {:0.6f}\n", test_mean_loss, test_accuracy);
    util::CSVLogger g_logger_testing((logs_path + "_test") + ".csv", "test_loss", "test_accuracy");
    g_logger_testing.log(test_mean_loss, test_accuracy);
    
#ifdef MEASURE_TIME
    fmt::print("training_time={}ms\n", util::unix_time() - training_time);
#endif
}
