#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <vector>
#include <random>
#include "image_manip.h"
#define TO_SIZE_T(x) ((x) >= 0 ? static_cast<size_t>(x) : 0)

namespace augumentation {
    enum class augumentation_type {
        none,
        flips,
        colors,
        crops,
        mixup,
        mixed,
    };

    template <typename Inner>
    class augumented_dataset : public torch::data::datasets::Dataset<augumented_dataset<Inner>> {
    public:
        augumented_dataset(Inner inner, augumentation_type type, int aug_number, bool aug_consistent) : inner(inner), type(type), aug_number(aug_number), aug_consistent(aug_consistent) {}

        torch::data::Example<> get(size_t index) override {
            switch (type) {
            case augumentation_type::none:
                return inner.get(index);
            case augumentation_type::flips: {
                auto inner_case = inner.get(index / 4);
                switch (index % 4) {
                case 0:
                    return inner_case;
                case 1:
                    return { image_manip::flip_x(inner_case.data), inner_case.target };
                case 2:
                    return { image_manip::flip_y(inner_case.data), inner_case.target };
                case 3:
                    return { image_manip::flip_y(image_manip::flip_x(inner_case.data)), inner_case.target };
                } 
            }
            case augumentation_type::colors: {
                auto inner_case = inner.get(index / 7);
                auto att = index % 7;
                if ((att % 2) == 1) { inner_case.data = image_manip::decolor(inner_case.data, 0); }
                if ((att / 2 % 2) == 1) { inner_case.data = image_manip::decolor(inner_case.data, 1); }
                if ((att / 4 % 2) == 1) { inner_case.data = image_manip::decolor(inner_case.data, 2); }
                return inner_case;
            }
            case augumentation_type::crops: {
                auto inner_case = inner.get(index / (aug_number + 1));
                auto w = inner_case.data.sizes()[1];
                auto h = inner_case.data.sizes()[2];
                auto x0 = rng_int(index * 4 + 0, w / 10);
                auto y0 = rng_int(index * 4 + 1, h / 10);
                auto x1 = x0 + (w - w / 10) + rng_int(index * 4 + 2, w / 10 - x0);
                auto y1 = y0 + (h - h / 10) + rng_int(index * 4 + 3, h / 10 - y0);
                return { image_manip::crop(inner_case.data, x0, x1, y0, y1), inner_case.target };
            }
            case augumentation_type::mixup: {

                auto i = index / (aug_number + 1);
                if (index % (aug_number + 1) == 0) { return inner.get(i); }
                auto j = rng_int(index * 2, *inner.size() - 2);
                if (TO_SIZE_T(j) >= i) { j += 1; }
                auto i_case = inner.get(i);
                auto j_case = inner.get(j);
                auto alpha = rng_float(index * 2 + 1);
                return { image_manip::mixup(i_case.data, j_case.data, alpha), i_case.target * alpha + j_case.target * (1 - alpha) };
            }
            case augumentation_type::mixed: {
                auto i = index / (aug_number + 1);
                if (index % (aug_number + 1) == 0) { return inner.get(i); }
                // Mixup + flip
                auto j = rng_int(index * 9, *inner.size() - 2);
                if (TO_SIZE_T(j) >= i) { j += 1; }
                auto i_case = inner.get(i);
                auto j_case = inner.get(j);
                if (rng_int(index * 9 + 1, 2) == 1) { i_case.data = image_manip::flip_x(i_case.data); }
                if (rng_int(index * 9 + 2, 2) == 1) { i_case.data = image_manip::flip_y(i_case.data); }
                if (rng_int(index * 9 + 3, 2) == 1) { j_case.data = image_manip::flip_x(j_case.data); }
                if (rng_int(index * 9 + 4, 2) == 1) { j_case.data = image_manip::flip_y(j_case.data); }
                auto alpha = rng_float(index * 9 + 5);
                auto ret_data = image_manip::mixup(i_case.data, j_case.data, alpha);
                auto ret_target = i_case.target * alpha + j_case.target * (1 - alpha);
                // Decolor
                auto att = rng_int(index * 9 + 6, 7);
                if ((att % 2) == 1) { ret_data = image_manip::decolor(ret_data, 0); }
                if ((att / 2 % 2) == 1) { ret_data = image_manip::decolor(ret_data, 1); }
                if ((att / 4 % 2) == 1) { ret_data = image_manip::decolor(ret_data, 2); }
                // Crop
                auto w = ret_data.sizes()[1];
                auto h = ret_data.sizes()[2];
                auto x0 = rng_int(index * 9 + 7, w / 10);
                auto y0 = rng_int(index * 9 + 8, h / 10);
                auto x1 = x0 + (w - w / 10) + rng_int(index * 13 + 11, w / 10 - x0);
                auto y1 = y0 + (h - h / 10) + rng_int(index * 13 + 12, h / 10 - y0);
                ret_data = image_manip::crop(ret_data, x0, x1, y0, y1);
                return {ret_data, ret_target};
            }
            }
            return inner.get(index);
        }
        
        torch::optional<size_t> size() const override {
            switch (type) {
            case augumentation_type::none:
                return *inner.size();
            case augumentation_type::flips:
                return *inner.size() * 4;
            case augumentation_type::colors:
                return *inner.size() * 7;
            case augumentation_type::crops:
                return *inner.size() * (aug_number + 1);
            case augumentation_type::mixup:
                return *inner.size() * (aug_number + 1);
            case augumentation_type::mixed:
                return *inner.size() * (aug_number + 1);
            }
            return *inner.size();
        }

    private:
        Inner inner;
        augumentation_type type;
        int aug_number;
        bool aug_consistent;
        int aug_index = 0;

        int rng_int(int seed, int max) {
            std::mt19937 gen(aug_consistent ? seed : aug_index++);
            std::uniform_int_distribution dist(0, max);
            return dist(gen);
        }

        int rng_float(int seed) {
            std::mt19937 gen(aug_consistent ? seed : aug_index++);
            std::uniform_real_distribution dist;
            return dist(gen);
        }
    };
}
