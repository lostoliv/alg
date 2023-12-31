#!/usr/bin/env python3
#
# Graph algorithms unit tests.
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


import random
import sys
import unittest

import graph
from graph import Graph


class TestGraph(unittest.TestCase):
  """
  Graph algorithms unit tests.
  """

  def test_graph_small(self):
    """
    Graph from:
https://www.geeksforgeeks.org/introduction-to-dijkstras-shortest-path-algorithm
    """
    g = Graph()
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 6)
    g.add_edge(1, 3, 5)
    g.add_edge(2, 3, 8)
    g.add_edge(3, 4, 10)
    g.add_edge(3, 5, 15)
    g.add_edge(4, 6, 2)
    g.add_edge(5, 6, 6)
    g.make_bidirectional()
    # Breadth-first search
    self.assertEqual(g.search_bfs(0), [0, 1, 2, 3, 4, 5, 6])
    # Depth-first search
    self.assertEqual(g.search_dfs(0), [0, 2, 3, 5, 6, 4, 1])
    # Cost from a node to another node in the graph
    start, end = 0, 6
    cost, path = g.cost_dijkstra(start, end)
    self.assertEqual(cost, 19)
    self.assertEqual(path, [0, 1, 3, 4, 6])
    self.assertEqual(path[0], start)
    self.assertEqual(path[-1], end)
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(start, end), cost)
    # Cost from a node to itself (should be 0)
    cost, path = g.cost_dijkstra(start, start)
    self.assertEqual(cost, 0)
    self.assertEqual(path, [start])
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(start, start), cost)

  def create_random_graph(seed, n, edge_min, edge_max, cost_min, cost_max,
                          node_prefix=None):
    """
    Create a random graph.

    Args:
      seed       : random seed
      n          : number of vertices in the graph
      edge_min   : minimal number of edge starting from each node
      edge_max   : maximal number of edge starting from each node
      cost_min   : minimal cost for each edge
      cost_max   : maximal cost for each edge
      node_prefix: prefix to name the nodes

    Returns:
      Random graph
    """
    random.seed(seed)
    g = Graph()
    for i in range(n):
      edge_count = random.randint(edge_min, edge_max + 1)
      for _ in range(edge_count):
        j = None
        while j is None:
          j = random.randint(0, n)
          if j == i:
            j = None
        cost = random.randint(cost_min, cost_max + 1)
        if node_prefix:
          # Use a prefix for the name of the node
          node_i = f'{node_prefix}_{i}'
          node_j = f'{node_prefix}_{j}'
        else:
          node_i = i
          node_j = j
        g.add_edge(node_i, node_j, cost)
    return g

  def create_random_path(seed, n):
    """
    Create a random path in the graph.

    Args:
      seed: random seed
      n   : number of vertices in the graph

    Returns:
      (start, end) path
    """
    random.seed(seed)
    start = random.randint(0, n)
    end = None
    while end is None:
      end = random.randint(0, n)
      if end == start:
        end = None
    return start, end

  def test_graph_medium(self):
    """
    Random medium graph.
    """
    seed = 1234
    n = 100
    g = TestGraph.create_random_graph(seed, n, 1, 2, 10, 100)
    start, end = TestGraph.create_random_path(seed + 1, n)
    cost, path = g.cost_dijkstra(start, end)
    self.assertEqual(cost, 364)
    self.assertEqual(path, [88, 40, 91, 37, 68, 83, 99, 71, 79, 54])
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(start, end), cost)

  def test_graph_large(self):
    """
    Random large graph.
    """
    seed = 5678
    n = 10000
    g = TestGraph.create_random_graph(seed, n, 2, 5, 10, 100)
    start, end = TestGraph.create_random_path(seed + 1, n)
    cost, path = g.cost_dijkstra(start, end)
    self.assertEqual(cost, 366)
    self.assertEqual(path, [4903, 2836, 2507, 2160, 3746, 6054, 2053, 4012,
                            2244, 8668, 3953, 684, 743])

  def test_graph_loop(self):
    """
    Loop.
    """
    g = Graph()
    n = 100
    # Each node i is only connected to the next (i + 1) node with a cost equal
    # to (i + 1)
    for i in range(n):
      g.add_edge(i, (i + 1) % n, i + 1)
    # There's only one way to go from 0 to (n - 1): we have to visit all the
    # nodes one after another
    cost, path = g.cost_dijkstra(0, n - 1)
    # Since we visit all the nodes and each node (i) has a cost of (i+1), the
    # expected cost will be: 1 + 2 + ... + n = n * (n - 1) / 2
    self.assertEqual(cost, n * (n - 1) // 2)
    # And the path will be all the nodes from 0 to n - 1
    self.assertEqual(path, list(range(100)))
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(0, n - 1), cost)
    # Make the graph bi-directional
    g.make_bidirectional()
    # Now we can cheat and just go backward for a cheaper cost
    cost, path = g.cost_dijkstra(0, n - 1)
    self.assertEqual(cost, n)
    self.assertEqual(path, [0, n - 1])
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(0, n - 1), cost)
    # Make the last edge (from the last node back to the first) prohibitively
    # expensive so that we will never use it
    # The optimal path will be the same one we had at the beginning: visiting
    # all the nodes one after another
    g.add_edge(0, n - 1, graph.INFINITY)
    cost, path = g.cost_dijkstra(0, n - 1)
    self.assertEqual(cost, n * (n - 1) // 2)
    self.assertEqual(path, list(range(100)))
    # Check Floyd-Warshall gives the same cost
    self.assertEqual(g.cost_floyd_warshall(0, n - 1), cost)


if __name__ == '__main__':
  unittest.main()
