ALGO_BASE = """
**For graphs with more than 18 nodes, approximate solution algorithms are preferred. On the contrary, for graphs with less than 18 nodes, the optimal solution is more suitable.**
{
  "graph_theory_problems": {
    "TSP": [
      {
        "algorithm": "TSP Greedy Algorithm",
        "solution_type": "Approximate",
        "description": "At each step, it chooses the shortest edge to build the path, which may not lead to the optimal solution.",
        "suitable_graph_size": "Suitable for large size graph, especially large graphs **(18 to 100 nodes)**. If the graph has **less than 18 nodes**, the algorithm to obtain the optimal solution is more suitable. The algorithm is fast and simple, with short computation times but does not guarantee the optimal solution.",
        "time_complexity": "O(n^2)",
        "input": {
          "adjacency_list": "A weighted graph represented as an adjacency matrix or edge list.",
        }
      }
    ],
    "GraphColoring": [
      {
        "algorithm": "Coloring Backtracking Algorithm",
        "solution_type": "Optimal",
        "description": "Uses depth-first search and backtracking to find the minimal coloring number.",
        "suitable_graph_size": "Suitable for small graphs with **fewer than 25 nodes**. For larger graphs, the time complexity becomes prohibitive.",
        "time_complexity": "O(2^n)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      },
      {
        "algorithm": "Greedy Coloring Algorithm",
        "solution_type": "Approximate",
        "description": "Colors each node based on its order, trying to minimize the number of colors, but it may not give the optimal solution.",
        "suitable_graph_size": "Suitable for large size graph, especially large graphs **(25 to 100 nodes)**, especially for larger graphs where the greedy approach can provide a good approximation quickly.",
        "time_complexity": "O(n^2)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      },
    ],
    "VertexCover":[
    {
        "algorithm": "VertexCover Brute Force Algorithm",
        "solution_type": "Optimal",
        "description": "Enumerates all possible subsets of vertices and checks each one to determine if it covers all edges. The smallest valid subset is the solution.",
        "suitable_graph_size": "Suitable for very small graphs with **fewer than 20 nodes** due to exponential time complexity.",
        "time_complexity": "O(2^n * n^2), where n is the number of vertices.",
        "input": {
          "adjacency_matrix": "An undirected graph represented as an adjacency matrix."
    },
    {
        "algorithm": "VertexCover Greedy Algorithm",
        "solution_type": "Approximation",
        "description": "Iteratively selects the vertex that covers the most uncovered edges until all edges are covered. The solution is not guaranteed to be optimal.",
        "suitable_graph_size": "Suitable for large size graph, especially large graphs **(20 to 100 nodes)**",
        "time_complexity": "O(n + m), where n is the number of vertices and m is the number of edges.",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency list."
    }
  },
    ],
    "CycleDetection": [
      {
        "algorithm": "Cycle Detection using Depth-First Search (DFS)",
        "solution_type": "Exact",
        "description": "Uses depth-first search (DFS) to traverse the graph while maintaining a recursion stack to detect back edges, which indicate a cycle.",
        "suitable_graph_size": "Efficient for graphs with **thousands of nodes**, as DFS runs in linear time.",
        "time_complexity": "O(n + m), where n is the number of vertices and m is the number of edges.",
        "input": {
          "adjacency_list": "A directed or undirected graph represented as an adjacency list."
        }
      }
    ],
    "Connectivity": [
      {
        "algorithm": "Connectivity Check using Depth-First Search (DFS)",
        "solution_type": "Exact",
        "description": "Uses Depth-First Search (DFS) to traverse the graph from a given start node, marking all reachable nodes. If the target node is reached during traversal, the two nodes are connected.",
        "suitable_graph_size": "Efficient for graphs with **thousands of nodes**, as DFS runs in linear time.",
        "time_complexity": "O(n + m), where n is the number of vertices and m is the number of edges.",
        "input": {
          "adjacency_list": "A directed or undirected graph represented as an adjacency list.",
          "start_node": "The node from which the search begins.",
          "end_node": "The node to check if it is reachable from the start node."
        }
      }
    ],
    "Shortest Path": [
      {
        "algorithm": "Shortest Path using Dijkstra's Algorithm",
        "solution_type": "Exact",
        "description": "Finds the shortest path and its total weight between two nodes in a weighted, undirected graph. The algorithm uses a priority queue to explore the graph, always expanding the node with the smallest known distance from the starting node.",
        "suitable_graph_size": "Efficient for graphs with **thousands of nodes** and positive edge weights, as Dijkstra's algorithm performs well with sparse graphs.",
        "time_complexity": "O((n + m) log n), where n is the number of vertices and m is the number of edges. The log factor stems from the use of a priority queue.",
        "input": {
          "adjacency_list":  "A directed or undirected graph represented as an adjacency list.",
          "start_node": "The node from which the shortest path computation begins.",
          "end_node": "The destination node to which the shortest path is computed."
        }
      }
    ]
  }
}    
"""