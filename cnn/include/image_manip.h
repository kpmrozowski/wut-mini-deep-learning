#include <torch/torch.h>
#include <opencv2/core.hpp>

namespace image_manip {
    cv::Mat torch_to_cv(torch::Tensor in);
    torch::Tensor cv_to_torch(cv::Mat in);
    torch::Tensor flip_x(torch::Tensor in);
    torch::Tensor flip_y(torch::Tensor in);
    torch::Tensor rotate(torch::Tensor in, float angle);
    torch::Tensor crop(torch::Tensor in, int x0, int x1, int y0, int y1);
    torch::Tensor mixup(torch::Tensor in0, torch::Tensor in1, float alphg);
} // namespace image_manip
