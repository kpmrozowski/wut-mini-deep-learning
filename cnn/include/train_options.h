#ifndef TRAIN_OPTIONS
#define TRAIN_OPTIONS
#include<tuple>
#include "augumentation.h"

#define CIFAR_PATH "/raid/cifar-10/"
#define LOGS_PATH "/raid/wut-mini-deep-learning/cnn/logs/"
#define MODELS_PATH "/raid/wut-mini-deep-learning/cnn/models/"

namespace regularization {

enum class regularization_type {
    none,
    l1,
    l2,
};

}

typedef std::tuple<
    regularization::regularization_type,
    double,
    augumentation::augumentation_type,
    std::string,
    int> SimulationSetting;



// class MyScheluder : public torch::optim::LRScheduler {
// public:

//     MyScheluder(torch::optim::Optimizer& optimizer,
//         const double base_lr)
//     : torch::optim::LRScheduler(optimizer)
//     , base_lr_(base_lr)
//     , current_lr_(base_lr) {}

//     double current_lr() { return current_lr_; }
//     void set_epoch(uint e) { epoch = e; }
// private:
//     std::vector<double> get_lrs() override {
//         current_lr_ = 0.36 * (std::atan(-(epoch - 10) / 4) + M_PI_2) * base_lr_;
//         auto new_lrs = this->get_current_lrs();
//         std::fill(new_lrs.begin(), new_lrs.end(), current_lr_);
//         return new_lrs;
//     }
//     const double base_lr_;
//     uint epoch = 0;
//     double current_lr_;
// };

std::vector<SimulationSetting> prepare_settings();

#endif // TRAIN_OPTIONS