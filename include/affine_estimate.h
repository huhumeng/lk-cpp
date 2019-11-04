#pragma once

#include "common.h"
#include <opencv2/core/core.hpp>


class ImageProcessor;

class AffineEstimator{

public:

    union AffineParameter
    {
        double data[6];

        struct{
            double p1, p2, p3, p4, p5, p6;
        };
    };
    

    AffineEstimator();
    ~AffineEstimator();

    // align a template image to source image using an affine transformation.
    void compute(const cv::Mat &source_image, const cv::Mat &template_image, const AffineParameter &affine_init, const Method &method = Method::kForwardAdditive);

private:

    // The Lucas-Kanade Algorithm
    // 1. Warp I with W(x; p) to compute I(W(x; p))
    // 2. Compute the error image I_warped - T
    // 3. Warped the gradient \Delta I with W(x; p). i.e compute the gradient of I_wraped
    // 4. Evaluate the Jacobian \frac{\paritial W}{\paritial p} at (x; p)
    // 5. Compute the steepest descent images
    // 6. Compute the Hessian matrix (J^T J)
    // 7. Compute the residual (J^T e)
    // 8. Compute \Delta p using Equation (10.)
    // 9. Update the parameter
    void computeFA();

    // The Compositional Algorithm
    // Approximately formula is minimized:
    // \sum_x \left [ I(W(W(x; \Delta p); p)) - T(x) \right ]^2
    // and with respect to \Delta p in each iteration updates the estimate of the wrap as:
    // $$W(x; p) \rightarrow W(x; p) \cdot W(x; \Delta p) $$
    
    void computeFC();

    ImageProcessor *image_processor_;
    
    bool debug_show_;

    cv::Mat tx_;
    cv::Mat imshow_;

    AffineParameter affine_;

    // maximum iteration number
    int max_iter_ ;

};