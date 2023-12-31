#!/usr/bin/env python3
#
# Graph algorithms.
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


import sys
from collections import defaultdict, deque


# Very large integer
INFINITY = sys.maxsize


class Graph:
  """
  Graph algorithms.
  """

  def __init__(self):
    """
    Initialize an empty graph.
    """
    self.graph = defaultdict(lambda: defaultdict(int))

  def add_edge(self, start, end, cost):
    """
    Add an edge between 2 nodes.

    Args:
      start: start node
      end  : end node
      cost : cost to go from start to end
    """
    self.graph[start][end] = cost

  def nodes(self):
    """
    Get the list of all the nodes in the graph.

    Returns:
      Name of all the nodes
    """
    nodes = set()
    for node in self.graph:
      nodes.add(node)
      for neighbor in self.graph[node]:
        nodes.add(neighbor)
    return nodes

  def make_bidirectional(self):
    """
    Make the graph bi-directional.
    For any edge(a, b) there will be an edge(b, a) with the same cost.
    Note that if edge(b, a) already exists, it will be replaced.
    """
    graph = self.graph.copy()
    for node in self.graph:
      for neighbor in self.graph[node]:
        # Given an existing (a, b) edge, cost(b --> a) = cost(a --> b)
        graph[neighbor][node] = self.graph[node][neighbor]
    self.graph = graph

  def merge(self, graph1, graph2):
    """
    Merge two graph.
    The two input graphes will not be modified.

    Args:
      graph1: first graph
      graph2: second graph
    """
    self.graph.__init__()
    self.graph.update(graph1.graph)
    self.graph.update(graph2.graph)

  def _search(self, start, bfs):
    """
    Graph search starting at a given node.

    Args:
      start: start node
      bfs  : breadth-first search? (if not, use deep-first search)

    Returns:
      List of nodes discovered from the search
    """

    if start not in self.graph:
      return None

    # Maintain a queue (or stack) and keep track of the nodes we visited
    visited = defaultdict(bool)
    queue = deque([start])
    visited[start] = True

    nodes = []
    while queue:
      if bfs:
        # Breadth-first search : we use a queue (first in, first out)
        current = queue.popleft()
      else:
        # Depth-first search: we use a stack (last in, first out)
        current = queue.pop()
      nodes.append(current)

      for neighbor in self.graph[current]:
        if visited[neighbor]:
          continue
        visited[neighbor] = True
        queue.append(neighbor)

    return nodes

  def search_bfs(self, start):
    """
    Breadth-first graph search starting at a given node.

    Args:
      start: start node

    Returns:
      List of nodes discovered from the search
    """
    return self._search(start, True)

  def search_dfs(self, start):
    """
    Deep-first graph search starting at a given node.

    Args:
      start: start node

    Returns:
      List of nodes discovered from the search
    """
    return self._search(start, False)

  def cost_map_dijkstra(self, start):
    """
    Calculate the cost map to go from a starting node to all other nodes
    in the graph using Dijkstra's algorithm.

    Args:
      start: start node

    Returns:
      Cost map
    """

    if start not in self.graph:
      return None

    # Maintain a list of unvisted nodes (aka all of them for now)
    unvisited = self.nodes()

    # Cost map
    cost_map = defaultdict(lambda: [INFINITY, None])
    # Initialize it for the start node with a zero cost
    cost_map[start] = [0, None]

    # Set the current node as the start node
    current = start

    while unvisited and current is not None:
      # Look at all the unvisited neighbors
      for neighbor in self.graph[current]:
        if neighbor not in unvisited:
          continue
        # Tentative cost = cost to the current node + cost to go to its
        # neighbor
        cost = cost_map[current][0] + self.graph[current][neighbor]
        # If the tentative cost is better than the current best cost, we update
        # it with the tentative cost
        if cost < cost_map[neighbor][0]:
          cost_map[neighbor] = [cost, current]

      # Mark the current node as visited (i.e. remove it from the list of
      # unvisited nodes)
      unvisited.remove(current)

      # Choose the next current node as the one with the smallest cost so far
      cost_min = INFINITY
      node_min = None
      for node in unvisited:
        if cost_map[node][0] < cost_min:
          cost_min = cost_map[node][0]
          node_min = node
      current = node_min

    return cost_map

  def cost_dijkstra(self, start, end, cost_map=None):
    """
    Calculate the cost to go from a starting node to another node in the
    graph using Dijkstra's algorithm.

    Args:
      start   : start node
      end     : end node
      cost_map: pre-computer map (optional)

    Returns:
      Cost to go from start to end, path to get there
    """

    # Get the cost map if missing
    if not cost_map:
      cost_map = self.cost_map_dijkstra(start)
      if not cost_map:
        return None, None
    if end not in cost_map:
      return None, None

    # Retrace the path from the end node back to the origin
    current = end
    path = [current]
    while cost_map[current][1] is not None:
      neighbor = cost_map[current][1]
      path.append(neighbor)
      current = neighbor

    # Return the cost at the end node and the path from start to end
    return cost_map[end][0], list(reversed(path))

  def cost_map_floyd_warshall(self):
    """
    Calculate the cost map to go from a any node to any other node in the
    graph (adjacency matrix) using Floyd-Warshall's algorithm.

    Returns:
      Cost map
    """

    # Get all the node and create a mapping to go from
    # (node name) --> index
    nodes = list(self.nodes())
    node_to_index = defaultdict(lambda: -1)
    for i, node in enumerate(nodes):
      node_to_index[node] = i
    n = len(nodes)

    # Cost adjacency matrix
    costs = [[INFINITY for _ in range(n)] for _ in range(n)]
    # Cost to go from any node to itself is zero
    for i in range(n):
      costs[i][i] = 0
    # Update the matrix with the costs in the graph
    for node in self.graph:
      for neighbor in self.graph[node]:
        node_idx = node_to_index[node]
        neighbor_idx = node_to_index[neighbor]
        costs[node_idx][neighbor_idx] = self.graph[node][neighbor]

    for k in range(n):
      for j in range(n):
        for i in range(n):
          # Tentative cost from i to k then k to j
          tentative_cost = costs[i][k] + costs[k][j]
          # If it's lower than the best one so far to go directly
          # from i to j, we update it
          if tentative_cost < costs[i][j]:
            costs[i][j] = tentative_cost

    return {'node_to_index': node_to_index, 'costs': costs}

  def cost_floyd_warshall(self, start, end, cost_map=None):
    """
    Calculate the cost to go from a starting node to another node in the
    graph using Floyd-Warshall's algorithm.

    Args:
      start   : start node
      end     : end node
      cost_map: pre-computer map (optional)

    Returns:
      Cost to go from start to end
    """

    # Get the cost map if missing
    if not cost_map:
      cost_map = self.cost_map_floyd_warshall()
      if not cost_map:
        return None

    # Get the node index using the mapping
    start = cost_map['node_to_index'][start]
    end = cost_map['node_to_index'][end]
    if start < 0 or end < 0:
      return None

    # Read the adjacency matrix
    return cost_map['costs'][start][end]
