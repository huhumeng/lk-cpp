#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "affine_estimate.h"
#include "image_process.h"

using namespace std;

int main(int argc, char **argv)
{

    if (argc != 2)
    {

        cout << "Usage: ./execute path_to_image" << endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], 0);

    AffineEstimator::AffineParameter affine_param;

    affine_param.p1 = 0;
    affine_param.p2 = 0.1;
    affine_param.p3 = 0;
    affine_param.p4 = 0;
    affine_param.p5 = 20;
    affine_param.p6 = 30;

    ImageProcessor *ip1 = new ImageProcessor;
    ImageProcessor *ip2 = new ImageProcessor;

    cv::Mat warped_image(100, 100, CV_64FC1);
    cv::Mat warped_gx(100, 100, CV_64FC1);
    cv::Mat warped_gy(100, 100, CV_64FC1);

    ip1->setInput(image);
    cv::Mat gx, gy;
    ip1->getGradient(gx, gy);

    for (int y = 0; y < 100; ++y)
    {
        for (int x = 0; x < 100; ++x)
        {
            double wx = (double)x * (1. + affine_param.p1) + (double)y * affine_param.p3 + affine_param.p5;
            double wy = (double)x * affine_param.p2 + (double)y * (1. + affine_param.p4) + affine_param.p6;

            warped_image.at<double>(y, x) = ip1->getBilinearInterpolation(wx, wy);

            warped_gx.at<double>(y, x) = ip1->getBilinearInterpolation(gx, wx, wy);
            warped_gy.at<double>(y, x) = ip1->getBilinearInterpolation(gy, wx, wy);
        }
    }

    ip2->setInput(warped_image);
    cv::Mat gx_w, gy_w;
    ip2->getGradient(gx_w, gy_w);

    std::cout << warped_gx - gx_w << std::endl;

    delete ip2;
    delete ip1;

    return 0;
}