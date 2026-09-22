// Stereo camera calibration from chessboard image pairs (e.g. a stereo endoscope).
//
//   ./stereo_calibrate --left data/calib/left --right data/calib/right
//                      --cols 9 --rows 6 --square 2.0 --out calib.yml
//
// Pairs are matched by sorted file name. --cols/--rows are *inner* corners,
// --square is the square size in mm, so the baseline and depth come out in mm.

#include "surgical_common.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

static const char* kKeys =
    "{help h |      | print help }"
    "{left   |      | folder with left images }"
    "{right  |      | folder with right images }"
    "{cols   | 9    | inner corners per row }"
    "{rows   | 6    | inner corners per column }"
    "{square | 1.0  | square size in mm }"
    "{out    | calib.yml | output calibration file }";

static std::vector<fs::path> listImages(const fs::path& dir) {
    std::vector<fs::path> v;
    for (auto& e : fs::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".bmp") v.push_back(e.path());
    }
    std::sort(v.begin(), v.end());
    return v;
}

static bool detect(const cv::Mat& gray, cv::Size pattern, std::vector<cv::Point2f>& corners) {
    if (!cv::findChessboardCorners(gray, pattern, corners,
                                   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE))
        return false;
    cv::cornerSubPix(gray, corners, {5, 5}, {-1, -1},
                     {cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1e-4});
    return true;
}

int main(int argc, char** argv) {
    surg::Args norm(argc, argv);
    cv::CommandLineParser args(norm.argc(), norm.data(), kKeys);
    if (args.has("help") || !args.has("left") || !args.has("right")) {
        args.printMessage();
        return args.has("help") ? 0 : 1;
    }
    const cv::Size pattern(args.get<int>("cols"), args.get<int>("rows"));
    const float square = args.get<float>("square");

    std::vector<cv::Point3f> board;
    for (int r = 0; r < pattern.height; ++r)
        for (int c = 0; c < pattern.width; ++c) board.emplace_back(c * square, r * square, 0.f);

    auto L = listImages(args.get<std::string>("left")), R = listImages(args.get<std::string>("right"));
    if (L.size() != R.size() || L.empty()) {
        std::cerr << "need the same, non-zero number of left/right images (" << L.size() << " vs " << R.size() << ")\n";
        return 1;
    }

    std::vector<std::vector<cv::Point3f>> obj;
    std::vector<std::vector<cv::Point2f>> ptsL, ptsR;
    cv::Size imgSize;
    for (size_t i = 0; i < L.size(); ++i) {
        cv::Mat gl = cv::imread(L[i].string(), cv::IMREAD_GRAYSCALE), gr = cv::imread(R[i].string(), cv::IMREAD_GRAYSCALE);
        imgSize = gl.size();
        std::vector<cv::Point2f> cl, cr;
        const bool ok = detect(gl, pattern, cl) && detect(gr, pattern, cr);
        std::cout << (ok ? "  ok   " : "  skip ") << L[i].filename().string() << "\n";
        if (ok) { obj.push_back(board); ptsL.push_back(cl); ptsR.push_back(cr); }
    }
    if (obj.size() < 5) {
        std::cerr << "only " << obj.size() << " usable pairs, need at least 5\n";
        return 1;
    }

    // 1) intrinsics per camera, 2) extrinsics with fixed intrinsics (more stable)
    cv::Mat K1, D1, K2, D2, R_, T, E, F;
    std::vector<cv::Mat> rv, tv;
    const double rmsL = cv::calibrateCamera(obj, ptsL, imgSize, K1, D1, rv, tv);
    const double rmsR = cv::calibrateCamera(obj, ptsR, imgSize, K2, D2, rv, tv);
    const double rmsS = cv::stereoCalibrate(obj, ptsL, ptsR, K1, D1, K2, D2, imgSize, R_, T, E, F,
                                            cv::CALIB_FIX_INTRINSIC,
                                            {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6});

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K2, D2, imgSize, R_, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0);

    cv::FileStorage fsOut(args.get<std::string>("out"), cv::FileStorage::WRITE);
    fsOut << "image_width" << imgSize.width << "image_height" << imgSize.height
          << "K1" << K1 << "D1" << D1 << "K2" << K2 << "D2" << D2 << "R" << R_ << "T" << T
          << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q
          << "rms_left" << rmsL << "rms_right" << rmsR << "rms_stereo" << rmsS;

    std::cout << std::fixed << std::setprecision(3) << "\nused pairs: " << obj.size() << "\n"
              << "reprojection RMS [px]  left " << rmsL << " | right " << rmsR << " | stereo " << rmsS << "\n"
              << "focal length [px]      " << K1.at<double>(0, 0) << "\n"
              << "baseline [mm]          " << cv::norm(T) << "\n"
              << "wrote " << args.get<std::string>("out") << "\n";
    return 0;
}
