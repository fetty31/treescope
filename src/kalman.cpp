/**
 * Implementation of KalmanFilter class.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

// Jonathan Lichtenfeld, 2023: Minor changes


/**
The MIT License (MIT)

Copyright (c) 2014 Hayk Martirosyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <stdexcept>
#include <ros/ros.h>

#include <treescope/kalman.h>

KalmanFilter::KalmanFilter(double dt, const Eigen::MatrixXd& A, const Eigen::MatrixXd& C, const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R, const Eigen::MatrixXd& P)
  : A(A), C(C), Q(Q), R(R), P0(P), m(C.rows()), n(A.rows()), dt(dt), initialized(false), I(n, n), x_hat(n), x_hat_new(n)
{
  I.setIdentity();
  //ROS_INFO("KalmanFilter constructed with A: [%ld x %ld]", A.rows(), A.cols());

}

KalmanFilter::KalmanFilter()
{
}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0)
{
  x_hat = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init()
{
  x_hat.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::predict(double dt, const Eigen::MatrixXd A)
{
  // ROS_INFO("Dimensions of A: [%ld x %ld]", A.rows(), A.cols());
  // ROS_INFO("Dimensions of x_hat: [%ld x %ld]", x_hat.rows(), x_hat.cols());
  // ROS_INFO("Dimensions of C: [%ld x %ld]", C.rows(), C.cols());
  if (!initialized)
    throw std::runtime_error("Filter is not initialized!");

  if (A.rows() != n || A.cols() != n) {
    ROS_ERROR("Dimension mismatch in A matrix: expected %dx%d, got %ldx%ld", n, n, A.rows(), A.cols());
    return;  // Early return to avoid further computation
  }

  // ROS_INFO("Predicting");
  // ROS_INFO("Dimensions of A: [%ld x %ld]", A.rows(), A.cols());
  // ROS_INFO("Dimensions of x_hat: [%ld x %ld]", x_hat.rows(), x_hat.cols());
  // ROS_INFO("Dimensions of C: [%ld x %ld]", C.rows(), C.cols());

  this->A = A;
  this->dt = dt;

  // ROS_INFO("Dimensions of A: [%ld x %ld]", A.rows(), A.cols());
  // ROS_INFO("Dimensions of x_hat: [%ld x %ld]", x_hat.rows(), x_hat.cols());
  // ROS_INFO("Dimensions of C: [%ld x %ld]", C.rows(), C.cols());


  x_hat_new = A * x_hat;
  P = A * P * A.transpose() + Q;
  x_hat = x_hat_new;

  t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y)
{
  if (!initialized)
    throw std::runtime_error("Filter is not initialized!");

  // ROS_INFO("Updating: Measurement Dimension [%ld]", y.size());
  // ROS_INFO("Dimensions of A: [%ld x %ld]", A.rows(), A.cols());
  // ROS_INFO("Dimensions of x_hat: [%ld x %ld]", x_hat.rows(), x_hat.cols());
  // ROS_INFO("Dimensions of C: [%ld x %ld]", C.rows(), C.cols());
 
  K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
  x_hat_new += K * (y - C * x_hat_new);
  P = (I - K * C) * P;
  x_hat = x_hat_new;
}