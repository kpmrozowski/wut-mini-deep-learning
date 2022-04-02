#pragma once

#include <torch/torch.h>

namespace image_manip {
    torch::Tensor flip_x(torch::Tensor in);
    torch::Tensor flip_y(torch::Tensor in);
    torch::Tensor decolor(torch::Tensor in, int channel);
    torch::Tensor crop(torch::Tensor in, int x0, int x1, int y0, int y1);
    torch::Tensor mixup(torch::Tensor in0, torch::Tensor in1, float alphg);
} // namespace image_manip
