#include <train_options.h>
#include <fmt/core.h>

std::vector<SimulationSetting> prepare_settings() {
    network_type network_type;
    regularization::regularization_type  regularization_type;
    double regularization_lambda;
    augumentation::augumentation_type augumentation_type;
    std::vector<SimulationSetting> settings;

    int experiment_type_idx = 26;
    std::vector<std::string> experiment_name{7, ""};
    fmt::print("\nfiles that are gonna be created:\n");
    for (int net_type_idx = 0; net_type_idx < 1; ++net_type_idx) {
        switch (net_type_idx) {
        case 0:
            network_type = network_type::convnet;
            experiment_name.at(0) = "REG_";
            break;
        case 1:
            network_type = network_type::mlp1;
            experiment_name.at(0) = "MLP1_REG_";
            break;
        case 2:
            network_type = network_type::mlpdeep;
            experiment_name.at(0) = "MLP_DEEP_REG_";
            break;
        }
        for (int reg_type_idx = 0; reg_type_idx < 1; ++reg_type_idx) {
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
            for (int reg_lambda_idx = 0; reg_lambda_idx < 1; ++reg_lambda_idx) {
            experiment_name.at(2) = "REGLAM_";
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
                if (regularization_type == regularization::regularization_type::none) {
                    experiment_name.at(3) = "none_";
                }
                for (int aug_type_idx = 0; aug_type_idx < 3; ++aug_type_idx) {
                    experiment_name.at(4) = "AUG_";
                    switch (aug_type_idx) { // Also this
                    case 0:
                        augumentation_type = augumentation::augumentation_type::none;
                        experiment_name.at(5) = "none_";
                        break;
                    case 1:
                        augumentation_type = augumentation::augumentation_type::crops;
                        experiment_name.at(5) = "crops_";
                        break;
                    case 2:
                        augumentation_type = augumentation::augumentation_type::colors;
                        experiment_name.at(5) = "colors_";
                        break;
                    case 3:
                        augumentation_type = augumentation::augumentation_type::flips;
                        experiment_name.at(5) = "flips_";
                        break;
                    case 4:
                        augumentation_type = augumentation::augumentation_type::mixup;
                        experiment_name.at(5) = "mixup_";
                        break;
                    case 5:
                        augumentation_type = augumentation::augumentation_type::mixed;
                        experiment_name.at(5) = "mixed_";
                        break;
                    }
                    // if (((regularization_type == regularization::regularization_type::l1
                    //         or regularization_type == regularization::regularization_type::none)
                    //     and regularization_lambda == 1e-4
                    //     and augumentation_type == augumentation::augumentation_type::none)
                    //     or augumentation_type == augumentation::augumentation_type::flips) {
                    //         continue;
                    //   }
                    std::string exp_name = "";
                    for (const auto& name_part : experiment_name) {
                        exp_name += name_part;
                    }
                    settings.emplace_back(
                        network_type,
                        regularization_type,
                        regularization_lambda,
                        augumentation_type,
                        exp_name,
                        experiment_type_idx);
                    fmt::print("EXP_{}_{}RUN_0-7.csv\n", experiment_type_idx, exp_name.c_str());
                    ++experiment_type_idx;
                }
            }
        }
    }
    return settings;
}
