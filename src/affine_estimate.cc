#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "affine_estimate.h"
#include "image_process.h"

std::array<cv::Point2d, 4> affinedRectangle(const AffineEstimator::AffineParameter &affine, const cv::Rect2d &rect)
{
    std::array<cv::Point2d, 4> result;

    result[0] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * rect.y + affine.p5,
                            affine.p2 * rect.x + (1 + affine.p4) * rect.y + affine.p6);

    result[1] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * (rect.y + rect.height) + affine.p5,
                            affine.p2 * rect.x + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

    result[2] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * (rect.y + rect.height) + affine.p5,
                            affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

    result[3] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * rect.y + affine.p5,
                            affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * rect.y + affine.p6);

    return result;
}

AffineEstimator::AffineEstimator() : max_iter_(80), debug_show_(true)
{
    image_processor_ = new ImageProcessor;
}

AffineEstimator::~AffineEstimator()
{
    if (image_processor_ != nullptr)
        delete image_processor_;
}

void AffineEstimator::compute(const cv::Mat &source_image, const cv::Mat &template_image, const AffineParameter &affine_init, const Method &method)
{

    memcpy(affine_.data, affine_init.data, sizeof(double) * 6);

    image_processor_->setInput(source_image);
    template_image.convertTo(tx_, CV_64FC1);

    if (debug_show_)
        cv::cvtColor(source_image, imshow_, cv::COLOR_GRAY2BGR);

    switch (method)
    {
    case Method::kForwardAdditive:
        computeFA();
        break;

    case Method::kForwardCompositional:
        computeFC();
        break;

    default:
        std::cerr << "Invalid method type, please check." << std::endl;
        break;
    }
}

void AffineEstimator::computeFA()
{
    Eigen::Map<Eigen::Matrix<double, 6, 1>> p(affine_.data);

    cv::Mat gx, gy;
    image_processor_->getGradient(gx, gy);

    int i = 0;
    for (; i < max_iter_; ++i)
    {

        if (debug_show_)
        {

            auto points = affinedRectangle(affine_, cv::Rect2d(0, 0, tx_.cols, tx_.rows));

            cv::Mat imshow = imshow_.clone();

            cv::line(imshow, points[0], points[1], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[1], points[2], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[2], points[3], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[3], points[0], cv::Scalar(0, 0, 255));

            cv::rectangle(imshow, cv::Rect(32, 52, 100, 100), cv::Scalar(0, 255, 0));

            cv::imshow("debug show", imshow);
            cv::waitKey(0);
        }

        Eigen::Matrix<double, 6, 6> hessian;
        hessian.setZero();

        Eigen::Matrix<double, 6, 1> residual;
        residual.setZero();

        double cost = 0.;

        for (int y = 0; y < tx_.rows; y++)
        {
            for (int x = 0; x < tx_.cols; x++)
            {
                double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
                double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

                if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
                    continue;

                double i_warped = image_processor_->getBilinearInterpolation(wx, wy);

                double err = i_warped - tx_.at<double>(y, x);

                double gx_wraped = image_processor_->getBilinearInterpolation(gx, wx, wy);
                double gy_wraped = image_processor_->getBilinearInterpolation(gy, wx, wy);

                Eigen::Matrix<double, 1, 6> jacobian;
                jacobian << x * gx_wraped, x * gy_wraped, y * gx_wraped, y * gy_wraped, gx_wraped, gy_wraped;

                hessian += jacobian.transpose() * jacobian;
                residual -= jacobian.transpose() * err;

                cost += err * err;
            }
        }

        Eigen::Matrix<double, 6, 1> delta_p = hessian.inverse() * residual;
        p += delta_p;

        std::cout << "Iteration " << i << " cost = " << cost << " squared delta p L2 norm = " << delta_p.squaredNorm() << std::endl;

        if (delta_p.squaredNorm() < 1e-12)
            break;
    }

    std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
              << affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
              << affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}

void AffineEstimator::computeFC()
{

    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> p(affine_.data);

    cv::Mat gx, gy;
    image_processor_->getGradient(gx, gy);

    int i = 0;
    for (; i < max_iter_; ++i)
    {

        if (debug_show_)
        {

            auto points = affinedRectangle(affine_, cv::Rect2d(0, 0, tx_.cols, tx_.rows));

            cv::Mat imshow = imshow_.clone();

            cv::line(imshow, points[0], points[1], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[1], points[2], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[2], points[3], cv::Scalar(0, 0, 255));
            cv::line(imshow, points[3], points[0], cv::Scalar(0, 0, 255));

            cv::rectangle(imshow, cv::Rect(32, 52, 100, 100), cv::Scalar(0, 255, 0));

            cv::imshow("debug show", imshow);
            cv::waitKey(0);
        }

        Eigen::Matrix<double, 6, 6> hessian;
        hessian.setZero();

        Eigen::Matrix<double, 6, 1> residual;
        residual.setZero();

        double cost = 0.;

        for (int y = 0; y < tx_.rows; y++)
        {
            for (int x = 0; x < tx_.cols; x++)
            {

                double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
                double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

                if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
                    continue;

                double i_warped = image_processor_->getBilinearInterpolation(wx, wy);

                double err = i_warped - tx_.at<double>(y, x);

                double gx_wraped = image_processor_->getBilinearInterpolation(gx, wx, wy);
                double gy_wraped = image_processor_->getBilinearInterpolation(gy, wx, wy);

                Eigen::Matrix<double, 1, 6> jacobian;
                
                Eigen::Matrix<double, 1, 2> j_I_x;
                j_I_x << gx_wraped, gy_wraped;

                Eigen::Matrix2d j_w_x;
                j_w_x << 1 + affine_.p1, affine_.p3, affine_.p2, 1 + affine_.p4;

                Eigen::Matrix<double, 2, 6> j_w_p;
                j_w_p.setZero();
                j_w_p(0, 0) = x;
                j_w_p(0, 2) = y;
                j_w_p(0, 4) = 1;
                j_w_p(1, 1) = x;
                j_w_p(1, 3) = y;
                j_w_p(1, 5) = 1;


                jacobian = j_I_x * j_w_x * j_w_p;

                hessian += jacobian.transpose() * jacobian;
                residual -= jacobian.transpose() * err;

                cost += err * err;
            }
        }

        double delta_p_data[9] = {0.};
        Eigen::Map<Eigen::Matrix<double, 6, 1>> delta_p(delta_p_data);

        delta_p = hessian.inverse() * residual;

        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Delta_p(delta_p_data);
        Delta_p(0, 0) += 1.;
        Delta_p(1, 1) += 1.;
        Delta_p(2, 2) = 1.;

        std::swap(p(0, 1), p(1, 0));
        std::swap(p(0, 1), p(1, 1));
        std::swap(p(0, 1), p(0, 2));

        p(0, 0) += 1.;
        p(1, 1) += 1.;
        
        // std::cout << p << std::endl;
        p *= Delta_p;

        p(0, 0) -= 1.;
        p(1, 1) -= 1.;

        std::swap(p(1, 0), p(0, 1));
        std::swap(p(1, 0), p(0, 2));
        std::swap(p(1, 0), p(1, 1));
        

        std::cout << "Iteration " << i << " cost = " << cost << " squared delta p L2 norm = " << delta_p.squaredNorm() << std::endl;

        if (delta_p.squaredNorm() < 1e-12)
            break;
    }

    std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
              << affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
              << affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}