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
        rotations,
        crops,
        mixup
    };

    template <typename Inner>
    class augumented_dataset : public torch::data::datasets::Dataset<augumented_dataset<Inner>> {
    public:
        augumented_dataset(Inner inner, augumentation_type type) : inner(inner), type(type) {}

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
            case augumentation_type::rotations: {
                auto inner_case = inner.get(index / 5);
                return { image_manip::rotate(inner_case.data, rng_int(index, 359)), inner_case.target };
            }
            case augumentation_type::crops: {
                auto inner_case = inner.get(index / 5);
                auto w = inner_case.data.sizes()[1];
                auto h = inner_case.data.sizes()[2];
                auto x0 = rng_int(index * 4 + 0, w / 2);
                auto y0 = rng_int(index * 4 + 1, h / 2);
                auto x1 = x0 + w / 2 + rng_int(index * 4 + 2, w - w / 2 - x0);
                auto y1 = y0 + h / 2 + rng_int(index * 4 + 3, h - h / 2 - y0);
                return { image_manip::crop(inner_case.data, x0, x1, y0, y1), inner_case.target };
            }
            case augumentation_type::mixup: {

                auto i = index / 5;
                if (index % 5 == 0) { return inner.get(i); }
                auto j = rng_int(index * 2, *inner.size() - 2);
                if (TO_SIZE_T(j) >= i) { j += 1; }
                auto i_case = inner.get(i);
                auto j_case = inner.get(j);
                auto alpha = rng_float(index * 2 + 1);
                return { image_manip::mixup(i_case.data, j_case.data, alpha), i_case.target * alpha + j_case.target * (1 - alpha) };
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
            case augumentation_type::rotations:
                return *inner.size() * 5;
            case augumentation_type::crops:
                return *inner.size() * 5;
            case augumentation_type::mixup:
                return *inner.size() * 5;
            }
            return *inner.size();
        }

    private:
        Inner inner;
        augumentation_type type;

        static int rng_int(int seed, int max) {
            std::mt19937 gen(seed);
            std::uniform_int_distribution dist(0, max);
            return dist(gen);
        }

        static int rng_float(int seed) {
            std::mt19937 gen(seed);
            std::uniform_real_distribution dist;
            return dist(gen);
        }
    };
}
