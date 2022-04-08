#include <image_io.h>
#include <torch/torch.h>
#include "image_manip.h"

namespace image_manip {
    torch::Tensor flip_x(torch::Tensor in) {
        return in.flip({1});
    }

    torch::Tensor flip_y(torch::Tensor in) {
        return in.flip({2});
    }

    torch::Tensor decolor(torch::Tensor in, int channel) {
        return in.index_put_({channel}, 0.5f);
    }

    torch::Tensor crop(torch::Tensor in, int x0, int x1, int y0, int y1) {
        auto cropped = in.slice(1, x0, x1).slice(2, y0, y1);
        return torch::nn::functional::interpolate(cropped.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<long> {in.size(1), in.size(2)})).squeeze();
    }

    torch::Tensor mixup(torch::Tensor in0, torch::Tensor in1, float alpha) {
        return in0 * alpha + in1 * (1 - alpha);
    }
} // namespace image_manip
