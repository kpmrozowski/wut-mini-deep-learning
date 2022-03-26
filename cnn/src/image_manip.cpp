#include <image_io.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace image_manip {
    // Heavily adapted from https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/6
    cv::Mat torch_to_cv(torch::Tensor in) {
        cv::Mat out(in.sizes()[1], in.sizes()[2], CV_8UC3);
        auto in_ = in * 255;
        std::memcpy((void *) out.data, at::_cast_Byte(in_).data_ptr(), sizeof(torch::kU8) * in.numel());
        return out;
    }
    torch::Tensor cv_to_torch(cv::Mat in) {
        in.convertTo(in, CV_32FC3, 1.0f / 255.0f);
        auto out = torch::from_blob(in.data, {3, in.cols, in.rows});
        out = out.clone();
        return out;
    }

    torch::Tensor flip_x(torch::Tensor in) {
        auto in_cv = torch_to_cv(in);
        cv::Mat out_cv;
        cv::flip(in_cv, out_cv, 1);
        return cv_to_torch(out_cv);
    }

    torch::Tensor flip_y(torch::Tensor in) {
        auto in_cv = torch_to_cv(in);
        cv::Mat out_cv;
        cv::flip(in_cv, out_cv, 0);
        return cv_to_torch(out_cv);
    }
} // namespace image_manip
