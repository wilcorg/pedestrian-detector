/* Roman Szlachtun, 272330
 *
 * Plik nagłówkowy jednokanałowego algorytmu odejmowania tła w C++.
 */

#pragma once
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <vector>

static const std::filesystem::path DATASET_DIR = "../../dataset";
static const std::filesystem::path PROJECT_ROOT_DIR = ".";

struct BoundingBox {
    int x1;
    int y1;
    int x2;
    int y2;
};

namespace fs = std::filesystem;

class Koller1 {
    cv::Mat diff_;
    cv::Mat background_;
    double deltaThreshold_;
    double alpha1_;
    double alpha2_;
    cv::Mat morphKernel_;
    double minAreaFraction_;

public:
    Koller1(double deltaThreshold, double alpha1, double alpha2, double minAreaFraction);

    /*
     * Inicjalizacja modelu tła poprzez obliczenie mediany klatek
     */
    void backgroundModelInit(const std::vector<fs::path> &frameList, double downscaleFactor);

    static cv::Mat readGrayscale(const fs::path &imgPath, double downscaleFactor);

    /*
     * Oblicza medianę dla każdego piksela z klatek
     */
    static cv::Mat imagesMedian(const std::vector<cv::Mat> &stack);

    /*
     * Zwraca wykryte ramki detekcji ruchu
     */
    std::vector<BoundingBox> predict(const cv::Mat &grayFrame);

    /*
     * Aktualizuje model tła oraz wylicza różnicę pomiędzy klatką a modelem tła
     */
    void kollerDiff(const cv::Mat &grayFrame);

    /*
     * Zwraca oczyszczoną różnicę pomiędzy klatką a modelem tła
     */
    cv::Mat getDiff() const;

    /*
     * Pozycje ramek detekcji są mnożone o downscaleFactor i nakładane na wejściową klatkę
     */
    static cv::Mat drawDetectionBoxes(const fs::path &imgPath, const std::vector<BoundingBox> &df, double downscaleFactor);

    /*
     * Przekształca kontury obszarów ruchu w prostokątne ramki
     */
    std::vector<BoundingBox> filterContours(const std::vector<std::vector<cv::Point>> &contours) const;
};
