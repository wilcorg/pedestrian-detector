/* Roman Szlachtun, 272330
 *
 * Program, wykrywający ludzi na obrazach za pomocą modelu SVM i cech HOG w C++.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>

using cv::Mat;
using cv::Size;
using cv::Rect;
using std::vector;

static vector<float> getSvmDetectorVector(const cv::Ptr<cv::ml::SVM>& svm)
{
    Mat sv = svm->getSupportVectors();
    if (sv.rows != 1) {
        sv = svm->getUncompressedSupportVectors();
    }

    Mat alpha, svidx;
    const double rho = svm->getDecisionFunction(0, alpha, svidx);

    const int svTotal = svm->getSupportVectors().rows;
    CV_Assert(svTotal == 1 && "Oczekiwano 1 wektora dla liniowego SVM");
    CV_Assert(alpha.total() == 1 && alpha.rows * alpha.cols == 1);
    CV_Assert(svidx.total() == 1 && svidx.rows * svidx.cols == 1);

    Mat sv32f;
    sv.convertTo(sv32f, CV_32F);

    vector<float> detector;
    detector.assign((float*)sv32f.ptr(), (float*)sv32f.ptr() + sv32f.cols);
    detector.push_back(static_cast<float>(-rho));
    return detector;
}

struct SvmConfig {
    Size windowSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
    int  nbins;
};

int main()
{
    cv::setUseOptimized(true);
    int startImg = 4050;
    int endImg = 4500;
    double downscaleFactor = 1.0;

    bool showDetections = false;
    const std::string svmPath = "/home/romka/CLionProjects/koller/svm/stage5/8-16-8-9-l2hys.cv2";
    // const std::string svmPath = "/home/romka/CLionProjects/koller/svm/stage5/16-32-8-9-l2hys.cv2";
    // const std::string svmPath = "/home/romka/CLionProjects/koller/svm/stage5/16-32-16-9-l2hys.cv2";

    SvmConfig cfg = {
            Size(64, 128),
            Size(16, 16),
            Size(8, 8),
            Size(8, 8),
            9
    };
//    SvmConfig cfg = {
//            Size(64, 128),
//            Size(32, 32),
//            Size(8, 8),
//            Size(16, 16),
//            9
//    };
//    SvmConfig cfg = {
//            Size(64, 128),
//            Size(32, 32),
//            Size(16, 16),
//            Size(16, 16),
//            9
//    };


    cv::HOGDescriptor hog(
            cfg.windowSize,
            cfg.blockSize,
            cfg.blockStride,
            cfg.cellSize,
            cfg.nbins
    );

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svmPath);
    if (svm.empty()) {
        std::cerr << "Nie udało się pobrać SVM ze ścieżki: " << svmPath << "\n";
        return 1;
    }
    hog.setSVMDetector(getSvmDetectorVector(svm));

    double total = 0.0;

    for (auto i = startImg; i <= endImg; i++) {
        std::string imgPath = "/home/romka/frames/" + std::to_string(i) + ".png";
        Mat gray = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        cv::resize(gray, gray, cv::Size(), 1.0 / downscaleFactor, 1.0 / downscaleFactor);

        vector<Rect> rects;
        vector<double> weights;
        auto detStart = std::chrono::steady_clock::now();
        hog.detectMultiScale(
                gray,
                rects,
                weights,
                0.3,
                cv::Size(8, 8),
                cv::Size(0, 0),
                1.05,
                1
        );
        auto detEnd = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(detEnd - detStart).count();

        std::cout << "Czas detekcji obrazu " << i << ": " << elapsed << " s" << std::endl;
        total += elapsed;

        if (showDetections) {
             vector<cv::Rect> boxes = rects;
             vector<float> scores(weights.begin(), weights.end());

             std::cout << "Detekcje: " << boxes.size() << "\n";
             for (size_t j = 0; j < boxes.size(); ++j) {
                 const Rect& r = boxes[j];
                 std::cout << " #" << j
                           << " box=[" << r.x << "," << r.y << "," << r.width << "," << r.height << "]"
                           << " score=" << scores[j] << "\n";
             }
        }
    }

    if (total != 0.0) {
        std::cout << "Średni czas detekcji na klatkę: " << (total / static_cast<double>(endImg - startImg + 1)) << "\n";
        std::cout << "Średni FPS: " << (1.0 / (total / static_cast<double>(endImg - startImg + 1))) << "\n";
    }

    return 0;
}
