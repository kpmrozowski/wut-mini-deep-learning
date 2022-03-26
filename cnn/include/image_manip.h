#include <torch/torch.h>
#include <opencv2/core.hpp>

namespace image_manip {
    cv::Mat torch_to_cv(torch::Tensor in);
    torch::Tensor cv_to_torch(cv::Mat in);
    torch::Tensor flip_x(torch::Tensor in);
    torch::Tensor flip_y(torch::Tensor in);
} // namespace image_manip
