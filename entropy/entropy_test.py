#!/usr/bin/env python3
#
# Entropy unit tests.
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


from enum import IntEnum
import unittest

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
import numpy as np


# A person is seating on a bridge above a highway in the USA.
# Over a week, this person is counting the cars passing under the bridge.
# For each car, this person is putting it in one of 4 buckets: is it a Japanese
# car, a Korean car, an American car or some other car.
#
# This person repeats the same experiment but this time in France. Of course
# the distribution of cars in France is widely different from the one in the
# USA.
#
# At the end of the week, this person is sending the observations in a binary
# message: "001110011100011111010100011..." etc.
#
# He needs to choose an encoding so the length of the message is as small as
# possible and the person receiving the message can decode it.
#
# For example, if this person observes 2 American cars, followed by 1 Japanese
# car, followed by 2 German cars (other), followed by 1 Japanese car then
# finally 1 Korean car, and this person is using as encoding (00 = Japan,
# 01 = Korea, 10 = USA, 11 = Other), then the message will be "10100011110001".
#
# Let's illustrate this problem and propose an encoding when this person is in
# the USA and then another one when this person is in France - tailored for the
# observed distribution of cars.
# Let's see how the length of each message is related to the entropy of the
# observed distributions.


# Cars country of origin (CCOO)
class CCOO(IntEnum):
  JAPAN = 0,
  KOREA = 1,
  USA = 2,
  OTHER = 3,
  COUNT = 4 # Number of classes

# Distribution of cars in the US = uniform distribution
DIST_USA = {
  CCOO.JAPAN: 0.25,
  CCOO.KOREA: 0.25,
  CCOO.USA: 0.25,
  CCOO.OTHER: 0.25
}

# Number of cars observed in 1 week
NUM_CARS_USA = 100000

# Proposed encoding of each cars, based on the COO.
# Each cars is encoded with 2 bits as we have a uniform distribution so we
# can't do better than this naive encoding.
# On average the per-car bit size will be:
#   2 * 0.25 + 2 * 0.25 + 2 * 0.25 + 2 * 0.25 = 2.0
# (which should be close to the entropy of the distribution, using log2)
ENCODING_USA = {
  CCOO.JAPAN: '00',
  CCOO.KOREA: '01',
  CCOO.USA: '10',
  CCOO.OTHER: '11'
}

# Distribution of cars in France
DIST_FR = {
  CCOO.JAPAN: 0.2,
  CCOO.KOREA: 0.1,
  CCOO.USA: 0.1,
  CCOO.OTHER: 0.6 # French cars?
}

# Number of cars observed in 1 week
NUM_CARS_FR = 10000

# Proposed encoding of each cars, based on COO.
# Naively we could use the same encoding as the USA - but can we do better?
# In a sense, there are a lot of "other" cars - could we represent "other" cars
# with a single bit? But then we have to use more than 1 bit for the rest of
# the cars which means that we'll have a different number of bits per car.
# Let's see how we could make this work if each car is encoded with 1, 2 or 3
# bits.
# If we read less than 3 bits (1 or 2) and it ends with 0, then we stop.
# For example if we need to read the message "100001011110", then the first car
# is encoded with "10" (we read at most 3 bits and stop when we find a 0).
# With the message "010001001010" we just read 1 bit (0).
# With the message "111111111111" we read 3 bits (111) and stop because we
# reached 3 bits.
# We encode with less bits the cars with the highest probability:
#   "Other" is encoded with 1 bit (0, finishing with a 0)
#   "Japan" is encoded with 2 bits (10, finishing with a 0)
#   "Korea" and "US", which are less frequent, are encoded with 3 bits
# (110 and 111 - does not contain a 0 before the 3rd bit so it won't stop
# before reading the 3 bits entirely).
# You can see that the encoding function is injective and we will always be
# able to decode any message.
# On average the per-car bit size will be:
#   2 * 0.2 + 3 * 0.1 + 3 * 0.1 + 1 * 0.6 = 1.6
# (which should be less than the naive encoding used for the USA and close to
# the entropy of the distribution, using log2)
ENCODING_FR = {
  CCOO.JAPAN: '10', # Second highest probability: use 2 bits
  CCOO.KOREA: '110', # Lower probability: use 3 bits
  CCOO.USA: '111', # Lower probability: use 3 bits
  CCOO.OTHER: '0' # Highest probability: use 1 bit
}


class TestEntropy(unittest.TestCase):
  """
  Entropy unit tests.
  """

  def test_cars(self):

    # Initialize the seed for reproducibility
    np.random.seed(1234)

    def _samples_cars(dist, num_samples):
      # Sample N cars given a distribution
      observations = np.zeros(num_samples, dtype=np.uint32)
      for i in range(num_samples):
        # For each observation, we tell which CCOO is seen
        observations[i] = np.random.choice(
          np.arange(CCOO.COUNT), p=[dist[x] for x in np.arange(CCOO.COUNT)])
      return observations

    def _dist(observations):
      # Calculate the distribution from observed samples
      dist = np.zeros(CCOO.COUNT, dtype=np.float64)
      unique, counts = np.unique(observations, return_counts=True)
      for ccoo in range(CCOO.COUNT):
        dist[unique[ccoo]] = counts[ccoo]
      return dist / np.sum(dist)

    def _entropy(x):
      # Entropy (log2) H(x): smallest number of bits we would need to encode
      # the distribution
      return -np.sum(x * np.log2(x))

    def _cross_entropy(x, y):
      # Cross-Entropy (log2) H(x,y): smallest number of bits we would need to
      # encode the distribution x with y
      return -np.sum(x * np.log2(y))

    def _kl_divergence(x, y):
      # Kullback–Leibler divergence (log2) KL(x, y): number of additional bits
      # we would need to encode x with y
      # KL(x, y) = H(x, y) - H(x)
      return np.sum(x * np.log2(x / y))

    # Observations in the USA and France over a week
    observations_usa = _samples_cars(DIST_USA, NUM_CARS_USA)
    observations_fr = _samples_cars(DIST_FR, NUM_CARS_FR)

    # Observed distributions
    dist_usa = _dist(observations_usa)
    dist_fr = _dist(observations_fr)

    def _encode(observations, encoding):
      # Encode the observations into a message using an encoding.
      # We encode the message as a string just for simplicity but we should
      # obviously use a bitarray instead.
      # Counting the number of bits in the message = calculating the length
      # of the string.
      msg = ''
      for x in observations:
        msg += encoding[x]
      return msg

    # Entropy of the CCOO distribution in the USA
    entropy_usa = _entropy(dist_usa)

    # Encode the observations in the USA with the USA encoding
    msg_usa_usa = _encode(observations_usa, ENCODING_USA)

    # Average bits per car with this encoding: it can't be smaller than the
    # entropy, but hopefully not too far from it
    bits_per_car_usa_usa = len(msg_usa_usa) / NUM_CARS_USA

    # Encode the observations in the USA with the FR encoding
    msg_usa_fr = _encode(observations_usa, ENCODING_FR)

    # Average bits per car with this encoding: it can't be smaller than the
    # entropy, and it should be higher than the one above with the USA encoding
    # (which is optimized for the distribution in the USA)
    bits_per_car_usa_fr = len(msg_usa_fr) / NUM_CARS_USA

    # Cross-entropy between USA and FR
    cross_entropy_usa_fr = _cross_entropy(dist_usa, dist_fr)

    # KL divergence between USA and FR
    kld_usa_fr = _kl_divergence(dist_usa, dist_fr)

    # To summarize we should have:
    #   entropy_usa < bits_per_car_usa_usa < bits_per_car_usa_fr
    self.assertLess(entropy_usa, bits_per_car_usa_usa)
    self.assertLess(bits_per_car_usa_usa, bits_per_car_usa_fr)

    # Entropy of the CCOO distribution in France.
    # Note that since the distribution is not uniform, the entropy will be
    # smaller than the one in the USA: the entropy is maximized for a uniform
    # distribution - which is why we should have, as an intuition, that we can
    # use a better encoding than the one we used in the USA.
    entropy_fr = _entropy(dist_fr)

    # Encode the observations in France with the FR encoding
    msg_fr_fr = _encode(observations_fr, ENCODING_FR)

    # Average bits per car with this encoding: it can't be smaller than the
    # entropy, but hopefully not too far from it
    bits_per_car_fr_fr = len(msg_fr_fr) / NUM_CARS_FR

    # Encode the observations in France with the USA encoding
    msg_fr_usa = _encode(observations_fr, ENCODING_USA)

    # Average bits per car with this encoding: it can't be smaller than the
    # entropy, and it should be higher than the one above with the FR encoding
    # (which is optimized for the distribution in France)
    bits_per_car_fr_usa = len(msg_fr_usa) / NUM_CARS_FR

    # Cross-entropy between FR and USA
    cross_entropy_fr_usa = _cross_entropy(dist_fr, dist_usa)

    # KL divergence between FR and USA
    kld_fr_usa = _kl_divergence(dist_fr, dist_usa)

    # To summarize we should have:
    #   entropy_fr < bits_per_car_fr_fr < bits_per_car_fr_usa
    self.assertLess(entropy_fr, bits_per_car_fr_fr)
    self.assertLess(bits_per_car_fr_fr, bits_per_car_fr_usa)

    print('USA:')
    print(f'  Entropy = {entropy_usa}')
    print(f'  Bit-per-car (USA encoding) = {bits_per_car_usa_usa}')
    print(f'  Bit-per-car (FR encoding) = {bits_per_car_usa_fr}')
    print(f'  Cross-Entropy (USA, FR) = {cross_entropy_usa_fr}')
    print(f'  KL divergence (USA, FR) = {kld_usa_fr}')
    print()
    print('France:')
    print(f'  Entropy = {entropy_fr}')
    print(f'  Bit-per-car (FR encoding) = {bits_per_car_fr_fr}')
    print(f'  Bit-per-car (USA encoding) = {bits_per_car_fr_usa}')
    print(f'  Cross-Entropy (FR, USA) = {cross_entropy_fr_usa}')
    print(f'  KL divergence (FR, USA) = {kld_fr_usa}')

    # Visualize the solution
    if plt is not None:
      labels = ['Entropy', 'BPC', 'BPC (alt)', 'Cross Entropy', 'KLD']
      width = 0.3
      data_usa = [entropy_usa, bits_per_car_usa_usa, bits_per_car_usa_fr,
                  cross_entropy_usa_fr, kld_usa_fr]
      data_fr = [entropy_fr, bits_per_car_fr_fr, bits_per_car_fr_usa,
                 cross_entropy_fr_usa, kld_fr_usa]
      plt.bar(np.arange((len(labels))), data_usa, width, label='USA')
      plt.bar(np.arange((len(labels))) + width, data_fr, width, label='France')
      plt.xticks(np.arange((len(labels))) + 0.5 * width, labels)
      plt.grid(linestyle='--', linewidth=1, axis='y', alpha=0.4)
      plt.legend(loc='best')
      plt.show()
    else:
      print('Install matplotlib to visualize the solution')


if __name__ == '__main__':
  unittest.main()
