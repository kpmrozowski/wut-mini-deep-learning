#include <image_io.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "image_manip.h"

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

    torch::Tensor rotate(torch::Tensor in, float angle) {
        auto in_cv = torch_to_cv(in);
        cv::Mat out_cv;
        int mid_x = in_cv.cols / 2.0;
        int mid_y = in_cv.rows / 2.0;
        auto mat = cv::getRotationMatrix2D(cv::Point2f(mid_x, mid_y), angle, 1.0);
        cv::warpAffine(in_cv, out_cv, mat, in_cv.size());
        return cv_to_torch(out_cv);
    }

    torch::Tensor crop(torch::Tensor in, int x0, int x1, int y0, int y1) {
        auto in_cv = torch_to_cv(in);
        auto cropped = in_cv(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        cv::Mat out_cv;
        std::cout << in_cv.size() << "\n";
        std::cout << cropped.size() << "\n";
        cv::resize(cropped, out_cv, in_cv.size(), cv::INTER_LINEAR);
        return cv_to_torch(out_cv);
    }
} // namespace image_manip
