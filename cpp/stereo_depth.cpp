// Dense depth from a stereo endoscope frame pair: rectification + Semi-Global Block Matching.
//
// With a calibration file from stereo_calibrate:
//   ./stereo_depth --left L.png --right R.png --calib calib.yml --out results/depth
// With already rectified images (e.g. Hamlyn rectified sequences):
//   ./stereo_depth --left L.png --right R.png --focal <f_px> --baseline <mm> --out results/depth
//
// Outputs: disparity.png (colour), depth_mm.png (16-bit, mm), cloud.ply (coloured point cloud)
// and, with --seg-model, per-class median depth (e.g. "how far is the grasper from the liver?").

#include "surgical_common.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

static const char* kKeys =
    "{help h     |      | print help }"
    "{left       |      | left image }"
    "{right      |      | right image }"
    "{calib      |      | calibration yml from stereo_calibrate (images get rectified) }"
    "{focal      | 0    | focal length in px (rectified input without --calib) }"
    "{baseline   | 0    | baseline in mm   (rectified input without --calib) }"
    "{num-disp   | 128  | disparity search range, multiple of 16 }"
    "{block      | 5    | SGBM block size (odd) }"
    "{min-depth  | 10   | discard points closer than this [mm] }"
    "{max-depth  | 300  | discard points farther than this [mm] }"
    "{seg-model  |      | optional ONNX segmentation model for per-class depth }"
    "{seg-width  | 448  | segmentation input width }"
    "{seg-height | 256  | segmentation input height }"
    "{out        | depth_out | output folder }";

static void writePly(const fs::path& path, const cv::Mat& xyz, const cv::Mat& colorBGR, const cv::Mat& valid) {
    std::vector<std::pair<cv::Vec3f, cv::Vec3b>> pts;
    for (int y = 0; y < xyz.rows; ++y)
        for (int x = 0; x < xyz.cols; ++x)
            if (valid.at<uchar>(y, x)) pts.emplace_back(xyz.at<cv::Vec3f>(y, x), colorBGR.at<cv::Vec3b>(y, x));
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
      << "\nproperty float x\nproperty float y\nproperty float z\n"
         "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (auto& [p, c] : pts)
        f << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << int(c[2]) << ' ' << int(c[1]) << ' ' << int(c[0]) << '\n';
}

int main(int argc, char** argv) {
    surg::Args norm(argc, argv);
    cv::CommandLineParser args(norm.argc(), norm.data(), kKeys);
    if (args.has("help") || !args.has("left") || !args.has("right")) {
        args.printMessage();
        return args.has("help") ? 0 : 1;
    }
    cv::Mat left = cv::imread(args.get<std::string>("left")), right = cv::imread(args.get<std::string>("right"));
    if (left.empty() || right.empty() || left.size() != right.size()) {
        std::cerr << "could not read a left/right pair of equal size\n";
        return 1;
    }

    // --- rectification ---------------------------------------------------------------
    cv::Mat Q;
    if (args.has("calib")) {
        cv::FileStorage fsIn(args.get<std::string>("calib"), cv::FileStorage::READ);
        cv::Mat K1, D1, K2, D2, R1, R2, P1, P2, m1x, m1y, m2x, m2y;
        fsIn["K1"] >> K1; fsIn["D1"] >> D1; fsIn["K2"] >> K2; fsIn["D2"] >> D2;
        fsIn["R1"] >> R1; fsIn["R2"] >> R2; fsIn["P1"] >> P1; fsIn["P2"] >> P2; fsIn["Q"] >> Q;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, left.size(), CV_32FC1, m1x, m1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, left.size(), CV_32FC1, m2x, m2y);
        cv::remap(left, left, m1x, m1y, cv::INTER_LINEAR);
        cv::remap(right, right, m2x, m2y, cv::INTER_LINEAR);
    } else {
        const double f = args.get<double>("focal"), B = args.get<double>("baseline");
        if (f <= 0 || B <= 0) {
            std::cerr << "give either --calib or --focal and --baseline\n";
            return 1;
        }
        const double cx = left.cols / 2.0, cy = left.rows / 2.0;
        Q = (cv::Mat_<double>(4, 4) << 1, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, f, 0, 0, 1.0 / B, 0);
    }

    // --- disparity (SGBM) ----------------------------------------------------------------
    const int block = args.get<int>("block"), nd = args.get<int>("num-disp");
    auto sgbm = cv::StereoSGBM::create(0, nd, block, 8 * 3 * block * block, 32 * 3 * block * block, 1, 63, 10,
                                       100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat gl, gr, disp16, disp;
    cv::cvtColor(left, gl, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, gr, cv::COLOR_BGR2GRAY);
    sgbm->compute(gl, gr, disp16);
    disp16.convertTo(disp, CV_32F, 1.0 / 16.0);

    // --- depth + point cloud ------------------------------------------------------------
    // Q maps (x, y, d) to (X, Y, Z, W); the sign convention depends on how it was built,
    // so depth is taken as |Z|.
    cv::Mat xyz;
    cv::reprojectImageTo3D(disp, xyz, Q, true);
    std::vector<cv::Mat> ch;
    cv::split(xyz, ch);
    cv::Mat depth = cv::abs(ch[2]);
    const double dmin = args.get<double>("min-depth"), dmax = args.get<double>("max-depth");
    cv::Mat valid = (disp > 0) & (depth > dmin) & (depth < dmax);

    const fs::path out = args.get<std::string>("out");
    fs::create_directories(out);
    cv::Mat depthMM, dispVis;
    depth.convertTo(depthMM, CV_16U);
    depthMM.setTo(0, ~valid);
    cv::imwrite((out / "depth_mm.png").string(), depthMM);
    cv::Mat disp8;
    cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8U, valid);
    cv::applyColorMap(disp8, dispVis, cv::COLORMAP_TURBO);
    dispVis.setTo(0, ~valid);
    cv::imwrite((out / "disparity.png").string(), dispVis);
    cv::imwrite((out / "left_rectified.png").string(), left);
    writePly(out / "cloud.ply", xyz, left, valid);

    std::vector<float> d;
    for (int y = 0; y < depth.rows; ++y)
        for (int x = 0; x < depth.cols; ++x)
            if (valid.at<uchar>(y, x)) d.push_back(depth.at<float>(y, x));
    std::cout << std::fixed << std::setprecision(1) << "valid pixels: " << 100.0 * d.size() / depth.total() << "%\n";
    if (!d.empty()) {
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        std::cout << "median depth: " << d[d.size() / 2] << " mm\n";
    }

    // --- optional: per-class depth from the segmentation model ----------------------------
    if (args.has("seg-model")) {
        auto net = cv::dnn::readNetFromONNX(args.get<std::string>("seg-model"));
        net.setInput(cv::dnn::blobFromImage(left, 1.0 / 255.0,
                                            {args.get<int>("seg-width"), args.get<int>("seg-height")},
                                            cv::Scalar(), true, false));
        cv::Mat labels, lab = surg::argmaxLabels(net.forward());
        cv::resize(lab, labels, left.size(), 0, 0, cv::INTER_NEAREST);
        cv::imwrite((out / "segmentation.png").string(), surg::overlay(left, labels, 0.5));
        std::cout << "per-class median depth:\n";
        for (int c = 1; c < static_cast<int>(surg::kClasses.size()); ++c) {
            std::vector<float> v;
            for (int y = 0; y < depth.rows; ++y)
                for (int x = 0; x < depth.cols; ++x)
                    if (labels.at<uchar>(y, x) == c && valid.at<uchar>(y, x)) v.push_back(depth.at<float>(y, x));
            if (v.size() < 200) continue;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            std::cout << std::setw(24) << surg::kClasses[c] << ": " << v[v.size() / 2] << " mm\n";
        }
    }
    std::cout << "wrote " << out << "\n";
    return 0;
}
