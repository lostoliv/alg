#!/usr/bin/env python3
#
# Newton and Gauss-Newton methods.
#
# The MIT License (MIT)
#
# Copyright (c)2023 Olivier Soares
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np


class Newton:
  """
  Newton method.

  Note that the gradient is approximated with numerical differentiation using
  the forward difference formula. It will lead to numerical instability that
  we won't cover here.
  """

  def __init__(self, f):
    """
    Initialization.

    Args:
      f: function f(x)
    """
    self.f = f

  def grad(self, x, dt, tol=np.finfo(np.float32).eps):
    """
    Approximate the gradient of the function at x.

    Args:
      x : where to evaluate the gradient
      dt: gradient step

    Returns:
      Approximate gradient
    """
    # df(x)/dx = lim(dt->0){ (f(x+dt) - f(x)) / dt }
    # Make sure dt is not too small to not get numerical instabilities
    y0 = self.f(x)
    y1 = self.f(x + dt)
    return (y1 - y0) / dt

  def zero(self, x, dt, max_iter=100000, tol=np.finfo(np.float32).eps):
    """
    Find the zero of the function starting at x.

    Args:
      x       : where to start
      dt      : gradient step
      max_iter: maximum number of iterations
      tol     : tolerance

    Returns:
      Zero of the function
    """
    for _ in range(max_iter):
      fx = self.f(x)
      if np.sum(fx**2) / x.size < tol:
        # Close enough from the solution
        break
      fx_dot = self.grad(x, dt)
      x = x - fx / fx_dot
    return x


class GaussNewton:
  """
  Gauss-Newton method.

  Note that the gradient is approximated with numerical differentiation using
  the forward difference formula. It will lead to numerical instability that
  we won't cover here.
  """

  def __init__(self, f):
    """
    Initialization.

    Args:
      f: function f(x)
    """
    self.f = f

  def grad(self, x, params, param, dt):
    """
    Approximate the gradient of the function at x relative to one of the
    parameters.

    Args:
      x     : where to evaluate the gradient
      params: list of function parameters
      param : parameter index on which we want to evaluate the gradient
      dt    : gradient step

    Returns:
      Approximate gradient
    """
    # Evaluate the function at x
    f = self.f(x, params)
    # Slightly nudge one of the parameters
    params[param] += dt
    # Evaluate the function at x
    f_dot = self.f(x, params)
    # Set the parameter to what it was
    params[param] -= dt
    # Forward difference formala to evaluate the gradient
    return (f_dot - f) / dt

  def jacobian(self, data, params, dt):
    """
    Approximate the Jacobian of the function.

    Args:
      data  : data points
      params: list of function parameters
      dt    : gradient step

    Returns:
      Approximate Jacobian
    """
    jf = np.zeros((data.shape[0], params.shape[0]))
    for j in range(params.shape[0]):
      for i in range(data.shape[0]):
        jf[i][j] = self.grad(data[i][0], params, j, dt[j])
    return jf

  def residuals(self, data, params):
    """
    Calculate the function residuals.

    Args:
      data  : data points
      params: list of function parameters

    Returns:
      Residuals
    """
    r = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
      # Residual = current optimized value - GT value
      r[i][0] = self.f(data[i][0], params) - data[i][1]
    return r

  @staticmethod
  def mse(y_pred, y_gt):
    """
    Mean-square error.

    Args:
      y_pred: prediction
      y_gt  : ground truth

    Returns:
      Mean-square error
    """
    return np.sum((y_pred - y_gt)**2) / y_pred.size

  def solve(self, data, params, dt, num_iter=100):
    """
    Gauss-Newton solver.

    Args:
      data    : data points
      params  : list of function parameters
      dt      : gradient step
      num_iter: number of iterations

    Returns:
      Updated parameters
    """

    for _ in range(num_iter):
      # For each step, calculate the Jacobian and residuals for each parameter
      jf = self.jacobian(data, params, dt)
      r = self.residuals(data, params)

      # Update the parameters
      jft =  jf.T
      jft_dot_jf_inv = np.linalg.inv(np.dot(jft, jf))
      params -= np.dot(np.dot(jft_dot_jf_inv, jft), r).flatten()

    return params
