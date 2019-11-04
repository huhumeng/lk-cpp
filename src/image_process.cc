#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "image_process.h"

ImageProcessor::ImageProcessor()
{
    kx_ = (cv::Mat_<double>(3, 3) << 0, 0, 0, -0.5, 0, 0.5, 0, 0, 0);
    ky_ = (cv::Mat_<double>(3, 3) << 0, -0.5, 0, 0, 0, 0, 0, 0.5, 0);
}

bool ImageProcessor::setInput(const cv::Mat &image)
{

    if(image.empty())
        return false;

    cv::Mat gray;

    // assert image is color bgr or gray
    if (image.type() != CV_8UC1)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = image;
    }

    gray.convertTo(image_, CV_64FC1);

    return true;
}

void ImageProcessor::getGradient(cv::Mat &gx, cv::Mat &gy) const
{

    if (image_.empty())
    {

        std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
        return;
    }

    cv::filter2D(image_, gx, -1, kx_);
    cv::filter2D(image_, gy, -1, ky_);
}

double ImageProcessor::getBilinearInterpolation(const cv::Mat &image, double x, double y)
{
    if (image.empty() || image.type() != CV_64FC1)
    {

        std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
        return -1;
    }

    int row = (int)y;
    int col = (int)x;

    double rr = y - row;
    double cc = x - col;

    return (1 - rr) * (1 - cc) * image.at<double>(row, col) +
           (1 - rr) * cc * image.at<double>(row, col + 1) +
           rr * (1 - cc) * image.at<double>(row + 1, col) +
           rr * cc * image.at<double>(row + 1, col + 1);
}

double ImageProcessor::getBilinearInterpolation(double x, double y) const
{
    return getBilinearInterpolation(image_, x, y);
}