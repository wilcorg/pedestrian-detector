/* Roman Szlachtun, 272330
 *
 * Kod, uruchamiający jednokanałowy algorytm odejmowania tła.
 */

#include <iostream>
#include <Koller1.h>
#include <opencv2/opencv.hpp>

int main() {
    cv::setUseOptimized(true);

    int startImg = 4050;
    int endImg = 4500;
    double downscaleFactor = 2.0;

    bool saveResults = false;
    bool saveMask = false;

    fs::path outPath = "/home/romka/CLionProjects/koller1/results/";

    Koller1 koller{
        15, 2e-3, 2e-4, 15e-4
    };

     // inicjalizacja modelu tła
    auto n = 20;
    std::vector<fs::path> initImgs(n);
    for (int i = 0; i < n; ++i) {
       initImgs[i] = fs::path("/home/romka/frames") / (std::to_string(startImg - 1 - (n - i - 1) * 25) + ".png");
    }
    koller.backgroundModelInit(initImgs, downscaleFactor);

    double total = 0.0;

    for (int i = startImg; i <= endImg; i++) {
        fs::path imgPath = fs::path("/home/romka/frames") / (std::to_string(i) + ".png");
        auto img = cv::imread(imgPath.string(), cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(), 1.0 / downscaleFactor, 1.0 / downscaleFactor);

        auto detStart = std::chrono::steady_clock::now();
        auto boxes = koller.predict(img);
        auto detEnd = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(detEnd - detStart).count();

        std::cout << "Czas detekcji obrazu " << i << ": " << elapsed << " s" << std::endl;
        total += elapsed;

        if (saveMask) {
            auto mask = koller.getDiff();
            cv::imwrite((outPath / (std::to_string(i) + "-mask.png")).string(), mask);
        }

        if (saveResults) {
            auto result = koller.drawDetectionBoxes(imgPath, boxes, downscaleFactor);
            cv::imwrite((outPath / (std::to_string(i) + "-result.png")).string(), result);
        }
    }

    if (total != 0.0) {
        std::cout << "Średni czas detekcji na klatkę: " << (total / static_cast<double>(endImg - startImg + 1)) << "\n";
        std::cout << "Średni FPS: " << (1.0 / (total / static_cast<double>(endImg - startImg + 1))) << "\n";
    }
}
