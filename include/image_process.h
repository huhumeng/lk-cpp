#pragma once

#include <opencv2/core/core.hpp>

class ImageProcessor{

public:

    ImageProcessor();

    ~ImageProcessor() = default;

    bool setInput(const cv::Mat &image);

    int width() const {return image_.cols;}
    int height() const {return image_.rows;}

    void getGradient(cv::Mat &gx, cv::Mat &gy) const;

    double getBilinearInterpolation(double x, double y) const;

    static double getBilinearInterpolation(const cv::Mat &image, double x, double y);


private:

    cv::Mat image_;
    cv::Mat kx_, ky_;
};