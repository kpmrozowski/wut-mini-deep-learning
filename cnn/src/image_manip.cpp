#include <image_io.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "image_manip.h"

namespace image_manip {
    // Heavily adapted from https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/6
    // Or at least were in the past, by now I'm not so sure.
    cv::Mat torch_to_cv(torch::Tensor in) {
        cv::Mat out(in.sizes()[1], in.sizes()[2], CV_32FC3);
        for (auto i = 0; i < in.sizes()[1]; ++i) {
            for (auto j = 0; j < in.sizes()[2]; ++j) {
                out.at<cv::Vec3f>(cv::Point(i, j)) = cv::Vec3f(in[0][i][j].item<float>(), in[1][i][j].item<float>(), in[2][i][j].item<float>());
            }
        }
        return out;
    }
    torch::Tensor cv_to_torch(cv::Mat in) {
        auto out = at::empty({3, in.cols, in.rows});
        for (auto i = 0; i < in.cols; ++i) {
            for (auto j = 0; j < in.rows; ++j) {
                auto p = in.at<cv::Vec3f>(cv::Point(i, j));
                out[0][i][j] = p[0];
                out[1][i][j] = p[1];
                out[2][i][j] = p[2];
            }
        }
        return out;
    }

    torch::Tensor flip_x(torch::Tensor in) {
        return in.flip({1});
    }

    torch::Tensor flip_y(torch::Tensor in) {
        return in.flip({2});
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
        auto cropped = in.slice(1, x0, x1).slice(2, y0, y1);
        return torch::nn::functional::interpolate(cropped.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<long> {in.size(1), in.size(2)})).squeeze();
    }

    torch::Tensor mixup(torch::Tensor in0, torch::Tensor in1, float alpha) {
        return in0 * alpha + in1 * (1 - alpha);
    }
} // namespace image_manip
