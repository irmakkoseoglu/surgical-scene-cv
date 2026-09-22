#pragma once
// Shared helpers for the C++ tools: class palette, argmax over network logits, overlays.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace surg {

// cv::CommandLineParser only understands "--key=value". This lets users also write
// the more common "--key value" form.
struct Args {
    std::vector<std::string> store;
    std::vector<const char*> argv;
    Args(int argc, char** raw) {
        store.emplace_back(raw[0]);
        for (int i = 1; i < argc; ++i) {
            std::string a = raw[i];
            const bool isKey = a.rfind("--", 0) == 0 && a.find('=') == std::string::npos;
            const bool nextIsValue = i + 1 < argc && !(raw[i + 1][0] == '-' && raw[i + 1][1] == '-');
            if (isKey && nextIsValue) a += std::string("=") + raw[++i];
            store.push_back(a);
        }
        for (auto& s : store) argv.push_back(s.c_str());
    }
    int argc() const { return static_cast<int>(argv.size()); }
    const char* const* data() const { return argv.data(); }
};

inline const std::vector<std::string> kClasses = {
    "Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat",
    "Grasper", "Connective Tissue", "Blood", "Cystic Duct", "L-hook Electrocautery",
    "Gallbladder", "Hepatic Vein", "Liver Ligament"};

// BGR colours, identical to the CholecSeg8k colour masks
inline const std::vector<cv::Vec3b> kColorsBGR = {
    {127, 127, 127}, {140, 140, 210}, {114, 114, 255}, {156, 70, 231}, {75, 183, 186},
    {0, 255, 170},   {0, 85, 255},    {0, 0, 255},     {0, 255, 255},  {184, 255, 169},
    {165, 160, 255}, {128, 50, 0},    {0, 74, 111}};

// logits: 4-D blob [1, C, H, W] -> CV_8U label map [H, W]
inline cv::Mat argmaxLabels(const cv::Mat& logits) {
    CV_Assert(logits.dims == 4 && logits.size[0] == 1);
    const int C = logits.size[1], H = logits.size[2], W = logits.size[3];
    const float* data = logits.ptr<float>();
    const size_t plane = static_cast<size_t>(H) * W;

    cv::Mat labels(H, W, CV_8U, cv::Scalar(0));
    cv::Mat best(H, W, CV_32F);
    std::copy(data, data + plane, best.ptr<float>());
    for (int c = 1; c < C; ++c) {
        const float* ch = data + c * plane;
        float* b = best.ptr<float>();
        uchar* l = labels.ptr<uchar>();
        for (size_t i = 0; i < plane; ++i) {
            if (ch[i] > b[i]) { b[i] = ch[i]; l[i] = static_cast<uchar>(c); }
        }
    }
    return labels;
}

inline cv::Mat colorize(const cv::Mat& labels) {
    cv::Mat out(labels.size(), CV_8UC3);
    labels.forEach<uchar>([&](const uchar& v, const int* pos) {
        out.at<cv::Vec3b>(pos[0], pos[1]) = v < kColorsBGR.size() ? kColorsBGR[v] : cv::Vec3b(0, 0, 0);
    });
    return out;
}

// Blend colour mask onto the frame; background (class 0) is left untouched.
inline cv::Mat overlay(const cv::Mat& frameBGR, const cv::Mat& labels, double alpha) {
    cv::Mat lab;
    cv::resize(labels, lab, frameBGR.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat color = colorize(lab), blended;
    cv::addWeighted(frameBGR, 1.0 - alpha, color, alpha, 0.0, blended);
    cv::Mat out = frameBGR.clone();
    blended.copyTo(out, lab != 0);
    return out;
}

// Share of pixels per class, e.g. for a simple "instrument visible" signal.
inline std::array<double, 13> classShares(const cv::Mat& labels) {
    std::array<double, 13> share{};
    const double n = static_cast<double>(labels.total());
    for (int c = 0; c < 13; ++c) share[c] = cv::countNonZero(labels == c) / n;
    return share;
}

}  // namespace surg
