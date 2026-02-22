/* Roman Szlachtun, 272330
 *
 * Impelementacja jednokanałowego algorytmu odejmowania tła w C++.
 *
 * Przyjmuje kolejne klatki nagrania, konwertuje ich do odcieni szarości
 * i zwraca ramki detekcji ruchu.
 */

#include <Koller1.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>

Koller1::Koller1(double deltaThreshold, double alpha1, double alpha2, double minAreaFraction)
    : deltaThreshold_(deltaThreshold),
      alpha1_(alpha1),
      alpha2_(alpha2),
      morphKernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))),
      minAreaFraction_(minAreaFraction)
{
}

std::vector<BoundingBox> Koller1::predict(const cv::Mat &grayFrame) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    kollerDiff(grayFrame);

    cv::findContours(diff_, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return filterContours(contours);
}

void Koller1::kollerDiff(const cv::Mat &grayFrame) {
    /*
     * Implementuje formułę selekcyjnej aktualizacji tła:
     * B_{t+1} = B_t + (alpha1*(1 - M_t) + alpha2*M_t) * D_t
     */
    if (background_.empty()) {
        throw std::runtime_error("Model tła nie jest zainicjalizowany.");
    }

    cv::Mat gray32f;
    grayFrame.convertTo(gray32f, CV_32F);

    cv::Mat Dt = gray32f - background_;
    cv::Mat absDt;
    cv::absdiff(gray32f, background_, absDt);

    // Binarna maska Mt odzwierciedla przekroczenie progu w Dt
    cv::Mat Mt_u8;
    cv::compare(absDt, deltaThreshold_, Mt_u8, cv::CMP_GE); // zakres 0 albo 255
    cv::Mat Mt;
    Mt_u8.convertTo(Mt, CV_32F, 1.0 / 255.0); // zakres 0.0 albo 1.0

    cv::Mat alpha_map = alpha1_ * (1.0f - Mt) + alpha2_ * Mt;
    background_ = background_ + alpha_map.mul(Dt);

    cv::min(background_, 255.0, background_);
    cv::max(background_, 0.0, background_);

    diff_ = Mt_u8;
    cv::morphologyEx(diff_, diff_, cv::MORPH_OPEN, morphKernel_, cv::Point(-1, -1), 1);
    cv::morphologyEx(diff_, diff_, cv::MORPH_CLOSE, morphKernel_, cv::Point(-1, -1), 2);
}

void Koller1::backgroundModelInit(const std::vector<fs::path> &frameList, double downscaleFactor) {
    std::vector<cv::Mat> images;

    for (const auto &p: frameList) {
        auto gray = readGrayscale(p.string(), downscaleFactor);
        gray.convertTo(gray, CV_32F);
        images.push_back(gray);
    }

    background_ = imagesMedian(images);
}

cv::Mat Koller1::drawDetectionBoxes(const fs::path &imgPath, const std::vector<BoundingBox> &df, double downscaleFactor) {
    auto image = cv::imread(imgPath.string(), cv::IMREAD_COLOR_BGR);
    for (const auto &b: df) {
        cv::rectangle(image, {static_cast<int>(downscaleFactor * b.x1), static_cast<int>(downscaleFactor * b.y1)}, {
                          static_cast<int>(downscaleFactor * b.x2), static_cast<int>(downscaleFactor * b.y2)
                      }, cv::Scalar(0, 0, 255), 4);
    }
    return image;
}

cv::Mat Koller1::getDiff() const {
    return diff_;
}


cv::Mat Koller1::imagesMedian(const std::vector<cv::Mat> &stack) {
    if (stack.empty()) {
        throw std::runtime_error("Nie można obliczyć mediany pustego stosu obrazów.");
    }

    const bool isEvenSizedStack = stack.size() % 2 == 0;
    const int rows = stack[0].rows;
    const int cols = stack[0].cols;

    for (const auto &image: stack) {
        if (image.type() != CV_32F || image.rows != rows || image.cols != cols) {
            throw std::runtime_error("Obrazy na stosie nie są spójne.");
        }
    }

    cv::Mat median(rows, cols, CV_32F);
    std::vector<float> valuesPerPixel;
    valuesPerPixel.reserve(stack.size());

    for (int y = 0; y < rows; ++y) {
        // tablica wskaźników na wiersz `y` dla każdego z obrazów na stosie
        auto rowYPointerForStackImages = std::make_unique<const float *[]>(stack.size());
        for (size_t i = 0; i < stack.size(); ++i) {
            // pobierz wskaźnik na wiersz `y` obrazu `i`
            rowYPointerForStackImages[i] = stack[i].ptr<float>(y);
        }

        auto *medianRowYPointer = median.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            valuesPerPixel.clear();
            for (size_t i = 0; i < stack.size(); ++i) {
                // dodajemy wartość piksela (x, y) z obrazu `i` na stosie
                valuesPerPixel.push_back(rowYPointerForStackImages[i][x]);
            }

            if (isEvenSizedStack) {
                auto midIt = valuesPerPixel.begin() + static_cast<long>(valuesPerPixel.size() / 2);
                // przestawia elementy w valuesPerPixel tak, że *midIt jest elementem, który by znalazł się na środku posortowanej tablicy
                // ale nie sortuje całej tablicy i działa w czasie liniowym
                std::ranges::nth_element(valuesPerPixel.begin(), midIt, valuesPerPixel.end());

                auto midIt2 = valuesPerPixel.begin() + static_cast<long>(valuesPerPixel.size() / 2 - 1);
                std::ranges::nth_element(valuesPerPixel.begin(), midIt2, valuesPerPixel.end());
                medianRowYPointer[x] = 0.5f * (*midIt + *midIt2);
            } else {
                auto midIt = valuesPerPixel.begin() + static_cast<long>(valuesPerPixel.size() / 2);
                std::ranges::nth_element(valuesPerPixel.begin(), midIt, valuesPerPixel.end());
                medianRowYPointer[x] = *midIt;
            }
        }
    }
    return median;
}

std::vector<BoundingBox> Koller1::filterContours(const std::vector<std::vector<cv::Point> > &contours) const {
    const int frameHeight = background_.rows;
    const int frameWidth = background_.cols;
    const int minAreaPx = std::max(1, static_cast<int>(minAreaFraction_ * frameHeight * frameWidth));

    std::vector<BoundingBox> out;
    for (const auto &c: contours) {
        double area = cv::contourArea(c);
        if (area < static_cast<double>(minAreaPx)) continue;

        cv::Rect r = cv::boundingRect(c);
        int x1 = std::max(0, r.x);
        int y1 = std::max(0, r.y);
        int x2 = std::min(frameWidth, r.x + r.width);
        int y2 = std::min(frameHeight, r.y + r.height);

        if (x2 > x1 && y2 > y1) {
            out.emplace_back(x1, y1, x2, y2);
        }
    }
    return out;
}

cv::Mat Koller1::readGrayscale(const fs::path &imgPath, double downscaleFactor) {
    cv::Mat gray = cv::imread(imgPath.string(), cv::IMREAD_GRAYSCALE);

    if (gray.empty()) throw std::runtime_error("Obraz nie jest znaleziony: " + imgPath.string());
    cv::resize(gray, gray, cv::Size(), 1.0 / downscaleFactor, 1.0 / downscaleFactor, cv::INTER_AREA);
    gray.convertTo(gray, CV_8U);
    return gray;
}
