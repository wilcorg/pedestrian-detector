/* Roman Szlachtun, 272330
 *
 * Impelementacja trójkanałowego algorytmu odejmowania tła w C++.
 *
 * Przyjmuje kolejne klatki nagrania, konwertuje ich do przestrzeni kolorów YUV
 * i zwraca ramki detekcji ruchu.
 */

#include <Koller3.h>

#include <algorithm>
#include <string>

Koller3::Koller3(double deltaLuma, double deltaChroma, double alpha1, double alpha2, double minAreaFraction)
    : deltaUV_(deltaChroma),
      deltaY_(deltaLuma),
      alpha1_(alpha1),
      alpha2_(alpha2),
      morphKernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))),
      minAreaFraction_(minAreaFraction)
{
}

std::vector<BoundingBox> Koller3::predict(const cv::Mat &bgrFrame) {
    kollerDiff(bgrFrame);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(diff_, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return filterContours(contours);
}

void Koller3::kollerDiff(const cv::Mat &bgrFrame) {
    /*
     * Implementuje formułę selekcyjnej aktualizacji tła:
     * B_{t+1} = B_t + (alpha1*(1 - M_t) + alpha2*M_t) * D_t
     */

    if (background_.empty()) {
        throw std::runtime_error("Model tła nie jest zainicjalizowany.");
    }

    cv::Mat yuv;
    cv::cvtColor(bgrFrame, yuv, cv::COLOR_BGR2YUV);
    yuv.convertTo(yuv, CV_32F);

    cv::Mat Dt = yuv - background_;

    // dzielenie obrazu na kanały Y, U, V
    std::vector<cv::Mat> dch(3);
    cv::split(Dt, dch);

    cv::Mat dY;
    cv::absdiff(dch[0], cv::Scalar(0), dY);
    cv::Mat dUV;

    // sqrt(U^2 + V^2)
    cv::magnitude(dch[1], dch[2], dUV);

    // maski kanałów Y i łączonych UV
    cv::Mat mUV, mY;

    // Binarna maska Mt odzwierciedla przekroczenie progu w Dt
    cv::Mat Mt_u8, Mt;
    // cv::compare zwraca zakresy 0 albo 255
    cv::compare(dUV, deltaUV_, mUV, cv::CMP_GE);
    cv::compare(dY, deltaY_, mY, cv::CMP_GE);

    cv::bitwise_or(mUV, mY, Mt_u8);
    Mt_u8.convertTo(Mt, CV_32F, 1.0 / 255.0); // zakres 0.0 albo 1.0

    cv::Mat alpha_map = alpha1_ * (1.0f - Mt) + alpha2_ * Mt;

    cv::Mat alpha3;
    cv::merge(std::vector{alpha_map, alpha_map, alpha_map}, alpha3);

    background_ = background_ + alpha3.mul(Dt);
    cv::min(background_, 255.0, background_);
    cv::max(background_, 0.0, background_);

    diff_ = Mt_u8;
    cv::morphologyEx(diff_, diff_, cv::MORPH_OPEN, morphKernel_, cv::Point(-1, -1), 1);
    cv::morphologyEx(diff_, diff_, cv::MORPH_CLOSE, morphKernel_, cv::Point(-1, -1), 2);
}

void Koller3::backgroundModelInit(const std::vector<fs::path> &frameList, double downscaleFactor) {
    std::vector<cv::Mat> images;

    for (const auto &p: frameList) {
        auto bgr = readBgr(p, downscaleFactor);

        cv::Mat yuv;
        cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);
        yuv.convertTo(yuv, CV_32F);
        images.push_back(yuv);
    }

    background_ = imagesMedian(images);
}

cv::Mat Koller3::drawDetectionBoxes(const fs::path &imgPath, const std::vector<BoundingBox> &df, double downscaleFactor) const {
    auto image = cv::imread(imgPath.string(), cv::IMREAD_COLOR_BGR);
    for (const auto &b: df) {
        cv::rectangle(image, {static_cast<int>(downscaleFactor * b.x1), static_cast<int>(downscaleFactor * b.y1)}, {
                static_cast<int>(downscaleFactor * b.x2), static_cast<int>(downscaleFactor * b.y2)
        }, cv::Scalar(0, 0, 255), 4);
    }
    return image;
}

cv::Mat Koller3::imagesMedian(const std::vector<cv::Mat> &stack) {
    if (stack.empty()) {
        throw std::runtime_error("Nie można obliczyć mediany pustego stosu obrazów.");
    }

    const bool isEvenSizedStack = stack.size() % 2 == 0;
    const int rows = stack[0].rows;
    const int cols = stack[0].cols;

    for (const auto &m: stack) {
        if (m.type() != CV_32FC3 || m.rows != rows || m.cols != cols)
            throw std::runtime_error("Obrazy na stosie nie są spójne.");
    }

    cv::Mat median(rows, cols, CV_32FC3);
    std::vector<float> valuesPerY, valuesPerU, valuesPerV;

    valuesPerY.reserve(stack.size());
    valuesPerU.reserve(stack.size());
    valuesPerV.reserve(stack.size());

    for (int y = 0; y < rows; ++y) {
        // tablica wskaźników na wiersz `y` dla każdego z obrazów na stosie
        auto rowYPointerForStackImages = std::make_unique<const cv::Vec3f *[]>(stack.size());
        for (size_t i = 0; i < stack.size(); ++i) {
            // pobierz wskaźnik na wiersz `y` obrazu `i`
            rowYPointerForStackImages[i] = stack[i].ptr<cv::Vec3f>(y);
        }

        auto *medianRowYPointer = median.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            valuesPerY.clear();
            valuesPerU.clear();
            valuesPerV.clear();
            for (size_t i = 0; i < stack.size(); ++i) {
                const cv::Vec3f v = rowYPointerForStackImages[i][x];
                valuesPerY.push_back(v[0]);
                valuesPerU.push_back(v[1]);
                valuesPerV.push_back(v[2]);
            }

            auto midY = valuesPerY.begin() + static_cast<long>(valuesPerY.size() / 2);
            auto midU = valuesPerU.begin() + static_cast<long>(valuesPerU.size() / 2);
            auto midV = valuesPerV.begin() + static_cast<long>(valuesPerV.size() / 2);
            std::ranges::nth_element(valuesPerY, midY);
            std::ranges::nth_element(valuesPerU, midU);
            std::ranges::nth_element(valuesPerV, midV);

            if (isEvenSizedStack) {
                auto midYb = valuesPerY.begin() + static_cast<long>(valuesPerY.size() / 2 - 1);
                auto midUb = valuesPerU.begin() + static_cast<long>(valuesPerU.size() / 2 - 1);
                auto midVb = valuesPerV.begin() + static_cast<long>(valuesPerV.size() / 2 - 1);
                std::ranges::nth_element(valuesPerY, midYb);
                std::ranges::nth_element(valuesPerU, midUb);
                std::ranges::nth_element(valuesPerV, midVb);
                medianRowYPointer[x][0] = 0.5f * (*midY + *midYb);
                medianRowYPointer[x][1] = 0.5f * (*midU + *midUb);
                medianRowYPointer[x][2] = 0.5f * (*midV + *midVb);
            } else {
                medianRowYPointer[x][0] = *midY;
                medianRowYPointer[x][1] = *midU;
                medianRowYPointer[x][2] = *midV;
            }
        }
    }
    return median;
}

std::vector<BoundingBox> Koller3::filterContours(const std::vector<std::vector<cv::Point> > &contours) {
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

cv::Mat Koller3::readBgr(const fs::path &imgPath, double downscaleFactor) {
    cv::Mat bgr = cv::imread(imgPath.string(), cv::IMREAD_COLOR_BGR);

    if (bgr.empty()) throw std::runtime_error("Obraz nie jest znaleziony: " + imgPath.string());
    cv::resize(bgr, bgr, cv::Size(), 1.0 / downscaleFactor, 1.0 / downscaleFactor, cv::INTER_AREA);
    return bgr;
}

cv::Mat Koller3::getDiff() const {
    return diff_;
}