#!/usr/bin/env python3
#
# Gradient-based optimization unit tests.
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
import torch


class TestGradient(unittest.TestCase):
  """
  Gradient-based optimization unit tests.

  Note that we won't develop our own array library and use PyTorch's instead.
  We could use Tensorflow, JAX, MLX, Tinygrad or any other array library.
  We will also not manually write the backward pass and rely on autodiff to
  keep track of the gradients.

  We use Pytorch in a very barebone way and avoid using torch.nn just to get
  back to the basics and re-implement a simple gradient descent algorithm
  around a simple toy example to try to learn a polynomial function.
  """

  def test_zero(self):

    # Initialize the seed for reproducibility
    np.random.seed(5678)
    torch.manual_seed(5678)

    class Model:
      def __init__(self, hidden_size):
        # Create a simple model with 2 layers and 1 activation
        self.w1 = torch.rand((1, hidden_size))
        self.b1 = torch.rand(hidden_size)
        self.w2 = torch.rand((hidden_size, 1))
        self.b2 = torch.rand(1)
        self.act = torch.tanh

      def copy(self):
        # Duplicate the model
        m = Model(self.w1.shape[1])
        for p_dst, p_src in zip(m.params(), self.params()):
          p_dst.data.copy_(p_src.data)
          p_dst.grad = None
        return m

      def __call__(self, x):
        # Inference
        h = self.act(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

      def params(self):
        # Returns all the model parameters
        return (self.w1, self.b1, self.w2, self.b2)

      def num_params(self):
        # Calculate the number of parameters in the model
        n_params = 0
        for p in self.params():
          n_params += p.numel()
        return n_params

      def requires_grad(self):
        # Flag the parameters to keep track of the gradients
        for p in self.params():
          p.requires_grad = True

      def zero(self):
        # Zero out the gradient for all parameters
        for p in self.params():
          p.grad = None

      def update(self, lr):
        # Update all the parameters based on its loss gradient
        for p in self.params():
          if p.grad is not None:
            p.data -= lr * p.grad

      @staticmethod
      def loss(y_pred, y_gt):
        # Loss = MSE
        return ((y_pred - y_gt) ** 2).mean()

      def train(self, data_train, data_val, lr, batch_size, num_iter):
        # Save the minimal validation error to save the best model (aka the
        # one that performs the best on the validation dataset)
        min_val_loss = np.finfo(np.float32).max
        model_best = self.copy()
        # Require all the parameters to keep track of the gradients
        self.requires_grad()
        for _ in range(num_iter):
          # Choose a training mini-batch
          batch = np.random.randint(0, data_train.shape[0], batch_size)
          x_train = torch.tensor(data_train[batch, 0],
            dtype=torch.float32).unsqueeze(dim=1)
          y_train_gt = torch.tensor(data_train[batch, 1],
            dtype=torch.float32).unsqueeze(dim=1)
          # Run the inference on the training data
          y_train_pred = self.__call__(x_train)
          # Calculate the training loss
          loss = self.loss(y_train_pred, y_train_gt)
          # Zero out the gradients
          self.zero()
          # Calculate the gradients of all the parameters relative to the
          # training loss
          loss.backward()
          with torch.no_grad():
            # Update the parameters based on the training loss gradients
            self.update(lr)
            # Calculate the validation loss
            # Choose a validation mini-batch
            batch = np.random.randint(0, data_val.shape[0], batch_size)
            x_val = torch.tensor(data_val[batch, 0],
              dtype=torch.float32).unsqueeze(dim=1)
            y_val_gt = torch.tensor(data_val[batch, 1],
              dtype=torch.float32).unsqueeze(dim=1)
            # Run the inference on the validation data
            y_val_pred = self.__call__(x_val)
            # Calculate the validation loss
            loss = self.loss(y_val_pred, y_val_gt)
            if loss.item() < min_val_loss:
              # New minimal validation loss: we save the model
              min_val_loss = loss.item()
              model_best = self.copy()
        # Return the best model
        return model_best

    def _f(x):
      # Ground truth function
      # We use the same one as in the Gauss-Newton unit test - we just try to
      # learn it using gradient descent
      return 1.5 + (2.5 * x) / (3.5 + x)

    # Generate noisy ground truth (x, y=f(x) + noise)
    noise = 0.2
    npoints = 10000
    x_min = 0
    x_max = 10
    data = np.zeros((npoints, 2))
    for i, x in enumerate(np.linspace(x_min, x_max, npoints)):
      y = _f(x)
      n = np.random.uniform(-noise, noise)
      data[i][0] = x
      data[i][1] = y + n

    # Split the dataset in 3: training (85%), validation (10%), test (5%)
    splits = [0.85, 0.95]
    indices = np.arange(npoints)
    np.random.shuffle(indices)
    data_train = data[indices[:int(splits[0] * npoints)],:]
    data_val = data[indices[int(splits[0] * npoints):
      int(splits[1] * npoints)],:]
    data_test = data[indices[int(splits[1] * npoints):],:]
    print(f'Train: {data_train.shape[0]} | '
          f'Val: {data_val.shape[0]} | '
          f'Test: {data_test.shape[0]}')

    # Train the model
    m = Model(hidden_size=4)
    print(f'Number of model parameters: {m.num_params()}')
    m = m.train(data_train, data_val, lr=0.05, batch_size=32, num_iter=1000)

    # Test the model using the test dataset
    data_pred = data_test.copy()
    with torch.no_grad():
      data_pred[:, 1] = m(torch.tensor(data_pred[:, 0],
        dtype=torch.float32).unsqueeze(dim=1)).squeeze().detach().numpy()
      loss = m.loss(data_pred[:, 1], data_test[:, 1])
      print(f'Testing loss: {loss:.5f}')
      self.assertLess(loss, 0.015)

    # Visualize the solution
    npoints_pred = 1000
    data_pred = np.zeros((npoints_pred, 2))
    data_pred[:, 0] = np.linspace(x_min, x_max, npoints_pred)
    with torch.no_grad():
      data_pred[:, 1] = m(torch.tensor(data_pred[:, 0],
        dtype=torch.float32).unsqueeze(dim=1)).squeeze().detach().numpy()
    if plt is not None:
      plt.scatter(data[:,0], data[:,1])
      plt.scatter(data_pred[:,0], data_pred[:,1])
      plt.show()
    else:
      print('Install matplotlib to visualize the solution')


if __name__ == '__main__':
  unittest.main()
