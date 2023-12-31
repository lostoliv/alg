#!/usr/bin/env python3
#
# Newton and Gauss-Newton unit tests.
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


import unittest

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
import numpy as np

from newton import Newton, GaussNewton


class TestNewton(unittest.TestCase):
  """
  Newton and Gauss-Newton unit tests.
  """

  def test_zero(self):
    """
    Find a zero of a function.
    """

    # Initialize the seed for reproducibility
    np.random.seed(1234)

    def _f(x):
      # Function x^3 - 3x + 1.
      return x**3 - 3*x + 1

    # Create the solver
    n = Newton(_f)

    # R --> R
    shape = (1,)

    # Start somewhere between [2, 10[
    x = np.random.uniform(2, 10, size=shape)

    # Gradient step
    dt = np.full(shape, 0.05)

    # Find the zero
    x = n.zero(x, dt)
    self.assertTrue(np.allclose(x, [1.5321], atol=0.0001))

  def test_fit(self):
    """
    Curve fitting.
    """

    # Initialize the seed for reproducibility
    np.random.seed(5678)

    def _f(x, params):
      # Function: a + (b * x) / (c + x)
      return params[0] + (params[1] * x) / (params[2] + x)

    # Use parameters a=1.5, b=2.5, c=3.5
    # Function: 1.5 + (2.5 * x) / (3.5 + x)
    params_gt = np.array([1.5, 2.5, 3.5])

    # Generate noisy ground truth (x, y=f(x) + noise)
    noise = 0.2
    npoints = 100
    x_min = 0
    x_max = 10
    data = np.zeros((npoints, 2))
    for i, x in enumerate(np.linspace(x_min, x_max, npoints)):
      y = _f(x, params_gt)
      n = np.random.uniform(-noise, noise)
      data[i][0] = x
      data[i][1] = y + n

    # Initial value for all parameters (use 1 for all parameters)
    nparams = params_gt.shape[0]
    params = np.ones(nparams)

    # Gradient step
    dt = np.full(nparams, 0.05)

    # Run the solver
    gn = GaussNewton(_f)
    params = gn.solve(data, params, dt)

    # Evaluate the error
    data_pred = np.zeros((npoints, 2))
    for i, x in enumerate(np.linspace(x_min, x_max, npoints)):
      y = _f(x, params)
      data_pred[i][0] = x
      data_pred[i][1] = y
    loss = GaussNewton.mse(data_pred[:,1], data[:,1])
    print(f'Loss: {loss:.5f}')
    self.assertLess(loss, 0.015)

    # Visualize the solution
    npoints_pred = 1000
    data_pred = np.zeros((npoints_pred, 2))
    for i, x in enumerate(np.linspace(x_min, x_max, npoints_pred)):
      y = _f(x, params)
      data_pred[i][0] = x
      data_pred[i][1] = y
    if plt is not None:
      plt.scatter(data[:,0], data[:,1])
      plt.scatter(data_pred[:,0], data_pred[:,1])
      plt.show()
    else:
      print('Install matplotlib to visualize the solution')


if __name__ == '__main__':
  unittest.main()
