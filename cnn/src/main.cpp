#include <train_options.h>
#include <Util/Concurrency.h>
#ifdef WITH_CUDA
#include <Eden_resources/Ngpus_Ncpus.h>
#endif

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
        std::string cifar_path{argc > 1 ? argv[1] : CIFAR_PATH};
        fmt::print("cifar path: {}\n", cifar_path);
        client_threads parallel_runs{num_gpus, setting, cifar_path};
        parallel_runs.join_clients();
    }

    std::cout << "Training finished!\n\n";
}
