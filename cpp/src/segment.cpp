// Real-time segmentation of laparoscopic images / videos with the exported ONNX model
// using OpenCV's DNN module (no Python or PyTorch needed at runtime).
//
//   ./segment --model models/unet.onnx --input video.mp4 --output out.mp4
//   ./segment --model models/unet.onnx --input frame.png --output overlay.png

#include "surgical_common.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static const char* kKeys =
    "{help h   |          | print help }"
    "{model    |          | path to ONNX model }"
    "{input    |          | image, video file or folder of images }"
    "{output   | out      | output image / video path or folder }"
    "{width    | 448      | network input width }"
    "{height   | 256      | network input height }"
    "{alpha    | 0.5      | overlay opacity }"
    "{threads  | 0        | OpenCV threads (0 = default) }";

class Segmenter {
public:
    Segmenter(const std::string& model, cv::Size size) : size_(size) {
        net_ = cv::dnn::readNetFromONNX(model);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // Returns label map at network resolution; latency in ms via out-param.
    cv::Mat run(const cv::Mat& frameBGR, double& ms) {
        const auto t0 = Clock::now();
        // RGB, scaled to [0,1]; ImageNet normalisation is part of the ONNX graph.
        cv::Mat blob = cv::dnn::blobFromImage(frameBGR, 1.0 / 255.0, size_, cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat labels = surg::argmaxLabels(net_.forward());
        ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return labels;
    }

private:
    cv::dnn::Net net_;
    cv::Size size_;
};

static bool isImage(const fs::path& p) {
    auto e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp";
}

static void printStats(const std::vector<double>& ms) {
    if (ms.empty()) return;
    std::vector<double> s(ms.begin() + (ms.size() > 5 ? 1 : 0), ms.end());  // skip warm-up
    std::sort(s.begin(), s.end());
    const double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    std::cout << std::fixed << std::setprecision(1) << "frames: " << ms.size() << " | mean " << mean
              << " ms | median " << s[s.size() / 2] << " ms | p95 " << s[static_cast<size_t>(0.95 * (s.size() - 1))]
              << " ms | " << 1000.0 / mean << " FPS\n";
}

int main(int argc, char** argv) {
    surg::Args norm(argc, argv);
    cv::CommandLineParser args(norm.argc(), norm.data(), kKeys);
    if (args.has("help") || !args.has("model") || !args.has("input")) {
        args.printMessage();
        return args.has("help") ? 0 : 1;
    }
    if (args.get<int>("threads") > 0) cv::setNumThreads(args.get<int>("threads"));

    const fs::path input = args.get<std::string>("input"), output = args.get<std::string>("output");
    const double alpha = args.get<double>("alpha");
    Segmenter seg(args.get<std::string>("model"), {args.get<int>("width"), args.get<int>("height")});
    std::vector<double> latencies;
    double ms = 0;

    try {
        if (fs::is_directory(input)) {  // folder of frames
            fs::create_directories(output);
            std::vector<fs::path> files;
            for (auto& e : fs::directory_iterator(input))
                if (isImage(e.path())) files.push_back(e.path());
            std::sort(files.begin(), files.end());
            for (auto& f : files) {
                cv::Mat img = cv::imread(f.string());
                cv::Mat lab = seg.run(img, ms);
                latencies.push_back(ms);
                cv::imwrite((output / f.filename()).string(), surg::overlay(img, lab, alpha));
            }
        } else if (isImage(input)) {  // single image
            cv::Mat img = cv::imread(input.string());
            if (img.empty()) throw std::runtime_error("cannot read " + input.string());
            cv::Mat lab = seg.run(img, ms);
            latencies.push_back(ms);
            cv::imwrite(output.string(), surg::overlay(img, lab, alpha));
            auto share = surg::classShares(lab);
            for (size_t c = 0; c < surg::kClasses.size(); ++c)
                if (share[c] > 0.001)
                    std::cout << std::setw(24) << surg::kClasses[c] << ": " << std::setprecision(1)
                              << std::fixed << 100 * share[c] << "%\n";
        } else {  // video
            cv::VideoCapture cap(input.string());
            if (!cap.isOpened()) throw std::runtime_error("cannot open " + input.string());
            const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
            cv::VideoWriter writer;
            cv::Mat frame;
            while (cap.read(frame)) {
                cv::Mat lab = seg.run(frame, ms);
                latencies.push_back(ms);
                cv::Mat vis = surg::overlay(frame, lab, alpha);
                cv::putText(vis, cv::format("%.1f ms", ms), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            {255, 255, 255}, 2);
                if (!writer.isOpened())
                    writer.open(output.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, vis.size());
                writer.write(vis);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    printStats(latencies);
    std::cout << "wrote " << output << "\n";
    return 0;
}
