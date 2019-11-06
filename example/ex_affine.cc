#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "affine_estimate.h"

using namespace std;

int main(int argc, char **argv)
{

    if (argc != 2)
    {

        cout << "Usage: ./execute path_to_image" << endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], 0);

    // Estimate the optical flow
    cv::Mat template_image = image(cv::Rect(32, 52, 100, 100)).clone();

    AffineEstimator::AffineParameter affine_param;
    
    // affine_param.p1 = 0.05;
    // affine_param.p2 = 0.1;
    // affine_param.p3 = 0.1;
    // affine_param.p4 = 0.1;
    // affine_param.p5 = 20;
    // affine_param.p6 = 30;

    affine_param.p1 = 0.1;
    affine_param.p2 = 0;
    affine_param.p3 = 0;
    affine_param.p4 = 0;
    affine_param.p5 = 30;
    affine_param.p6 = 50;

    AffineEstimator estimator;
    estimator.compute(image, template_image, affine_param, Method::kForwardCompositional);

    return 0;
}