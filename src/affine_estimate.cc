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

void AffineEstimator::debugShow()
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

    case Method::kBackwardAdditive:
        computeBA();
        break;

    case Method::kBackwardCompositional:
        computeBC();
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
            debugShow();

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

                double gx_warped = image_processor_->getBilinearInterpolation(gx, wx, wy);
                double gy_warped = image_processor_->getBilinearInterpolation(gy, wx, wy);

                Eigen::Matrix<double, 1, 6> jacobian;
                jacobian << x * gx_warped, x * gy_warped, y * gx_warped, y * gy_warped, gx_warped, gy_warped;

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
    Eigen::Map<Eigen::Matrix<double, 6, 1>> p(affine_.data);

    int i = 0;
    for (; i < max_iter_; ++i)
    {

        if (debug_show_)
            debugShow();

        Eigen::Matrix<double, 6, 6> hessian;
        hessian.setZero();

        Eigen::Matrix<double, 6, 1> residual;
        residual.setZero();

        double cost = 0.;

        cv::Mat warped_i(tx_.size(), CV_64FC1);

        for (int y = 0; y < tx_.rows; y++)
        {
            for (int x = 0; x < tx_.cols; x++)
            {

                double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
                double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

                if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
                {
                    warped_i.at<double>(y, x) = 0;
                    continue;
                }

                warped_i.at<double>(y, x) = image_processor_->getBilinearInterpolation(wx, wy);
            }
        }

        ImageProcessor temp_proc;
        temp_proc.setInput(warped_i);

        cv::Mat warped_gx, warped_gy;
        temp_proc.getGradient(warped_gx, warped_gy);

        for (int y = 0; y < tx_.rows; y++)
        {
            for (int x = 0; x < tx_.cols; x++)
            {
                double err = warped_i.at<double>(y, x) - tx_.at<double>(y, x);

                Eigen::Matrix<double, 1, 6> jacobian;

                double gx_warped = warped_gx.at<double>(y, x);
                double gy_warped = warped_gy.at<double>(y, x);

                jacobian << x * gx_warped, x * gy_warped, y * gx_warped, y * gy_warped, gx_warped, gy_warped;

                hessian += jacobian.transpose() * jacobian;
                residual -= jacobian.transpose() * err;

                cost += err * err;
            }
        }

        double delta_p_data[6] = {0.};
        Eigen::Map<Eigen::Matrix<double, 6, 1>> delta_p(delta_p_data);

        delta_p = hessian.inverse() * residual;

        double inc[6] = {0.};
        memcpy(inc, delta_p_data, sizeof(double) * 6);
        inc[0] += (affine_.p1 * delta_p_data[0] + affine_.p3 * delta_p_data[1]);
        inc[1] += (affine_.p2 * delta_p_data[0] + affine_.p4 * delta_p_data[1]);
        inc[2] += (affine_.p1 * delta_p_data[2] + affine_.p3 * delta_p_data[3]);
        inc[3] += (affine_.p2 * delta_p_data[2] + affine_.p4 * delta_p_data[3]);
        inc[4] += (affine_.p1 * delta_p_data[4] + affine_.p3 * delta_p_data[5]);
        inc[5] += (affine_.p2 * delta_p_data[4] + affine_.p4 * delta_p_data[5]);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> increment(inc);
        p += increment;

        std::cout << "Iteration " << i << " cost = " << cost << " squared delta p L2 norm = " << delta_p.squaredNorm() << std::endl;

        if (delta_p.squaredNorm() < 1e-12)
            break;
    }

    std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
              << affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
              << affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}

void AffineEstimator::computeBA()
{
}

void AffineEstimator::computeBC()
{
    ImageProcessor temp_proc;
    temp_proc.setInput(tx_);

    // Pre-compute
    cv::Mat gx, gy;
    temp_proc.getGradient(gx, gy);

    cv::Mat xgx(gx.size(), gx.type());
    cv::Mat xgy(gx.size(), gx.type());
    cv::Mat ygx(gx.size(), gx.type());
    cv::Mat ygy(gx.size(), gx.type());

    Eigen::Matrix<double, 6, 6> hessian;
    hessian.setZero();

    for (int y = 0; y < tx_.rows; y++)
    {
        for (int x = 0; x < tx_.cols; x++)
        {
            xgx.at<double>(y, x) = x * gx.at<double>(y, x);
            xgy.at<double>(y, x) = x * gy.at<double>(y, x);
            ygx.at<double>(y, x) = y * gx.at<double>(y, x);
            ygy.at<double>(y, x) = y * gy.at<double>(y, x);

            Eigen::Matrix<double, 1, 6> jacobian;

            jacobian << xgx.at<double>(y, x), xgy.at<double>(y, x), ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x);

            hessian += jacobian.transpose() * jacobian;
        }
    }

    std::cout << hessian << std::endl;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; ++j)
        {
            if (i != j)
                hessian(i, j) = 0;            
        }
    }
    
    Eigen::Matrix<double, 6, 6> H_inv = hessian.inverse();

    int i = 0;
    for (; i < max_iter_; ++i)
    {

        if (debug_show_)
            debugShow();

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

                double err = tx_.at<double>(y, x) - image_processor_->getBilinearInterpolation(wx, wy);

                Eigen::Matrix<double, 1, 6> jacobian;
                jacobian << xgx.at<double>(y, x), xgy.at<double>(y, x), ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x);

                residual -= jacobian.transpose() * err;

                cost += err * err;
            }
        }

        Eigen::Matrix<double, 6, 1> delta_p = H_inv * residual;

        Eigen::Matrix3d delta_m, warp_m;

        warp_m << 1 + affine_.p1, affine_.p3, affine_.p5, affine_.p2, 1 + affine_.p4, affine_.p6, 0, 0, 1;
        delta_m << 1 + delta_p(0), delta_p(2), delta_p(4), delta_p(1), 1 + delta_p(3), delta_p(5), 0, 0, 1;

        Eigen::Matrix3d new_warp = warp_m * (delta_m.inverse());
        affine_.p1 = new_warp(0, 0) - 1.;
        affine_.p2 = new_warp(1, 0);
        affine_.p3 = new_warp(0, 1);
        affine_.p4 = new_warp(1, 1) - 1.;
        affine_.p5 = new_warp(0, 2);
        affine_.p6 = new_warp(1, 2);

        std::cout << "Iteration " << i << " cost = " << cost << " squared delta p L2 norm = " << delta_p.squaredNorm() << std::endl;

        if (delta_p.squaredNorm() < 1e-12)
            break;
    }

    std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
              << affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
              << affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}