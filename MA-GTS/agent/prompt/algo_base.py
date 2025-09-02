ALGO_BASE = """
{
  "graph_theory_problems": {
    "TSP": [
      {
        "algorithm": "Brute Force",
        "solution_type": "Optimal",
        "description": "By exhaustively checking all possible paths, it finds the shortest route.",
        "suitable_graph_size": "Suitable for small graphs (up to 10 nodes) due to factorial time complexity (O(n!)), as the computation time increases drastically with more nodes.",
        "time_complexity": "O(n!)",
        "input": {
          "adjacency_list": "A complete weighted graph represented as an adjacency matrix or edge list.",
          "start_node": "The starting node for the traveling salesman problem."
        }
      },
      {
        "algorithm": "Dynamic Programming (Held-Karp Algorithm)",
        "solution_type": "Optimal",
        "description": "Uses dynamic programming to reduce repeated calculations, building the global solution from subproblems.",
        "suitable_graph_size": "Suitable for medium-sized graphs (up to 15 nodes). This algorithm has higher time complexity, so it’s more suitable for smaller to medium-sized instances.",
        "time_complexity": "O(n^2 * 2^n)",
        "input": {
          "adjacency_list: "A complete weighted graph represented as an adjacency matrix or edge list.",
          "start_node": "The starting node for the traveling salesman problem."
        }
      },
      {
        "algorithm": "Greedy Algorithm",
        "solution_type": "Approximate",
        "description": "At each step, it chooses the shortest edge to build the path, which may not lead to the optimal solution.",
        "suitable_graph_size": "Suitable for any size graph, especially large graphs (over hundreds of nodes). The algorithm is fast and simple, with short computation times but does not guarantee the optimal solution.",
        "time_complexity": "O(n^2)",
        "input": {
          "adjacency_list": "A weighted graph represented as an adjacency matrix or edge list.",
          "start_node": "The starting node for the traveling salesman problem."
        }
      },
      {
        "algorithm": "Genetic Algorithm",
        "solution_type": "Approximate",
        "description": "Simulates the process of natural selection to search for the optimal solution, suitable for large-scale problems.",
        "suitable_graph_size": "Suitable for large graphs, with up to hundreds of nodes. The time complexity is moderate, but it depends on parameter adjustments.",
        "time_complexity": "O(n^2 * generations)",
        "input": {
          "adjacency_list": "A complete weighted graph represented as an adjacency matrix or edge list.",
          "start_node": "The starting node for the traveling salesman problem.",
          "population_size": "The number of individuals in the population.",
          "mutation_rate": "The rate at which mutations occur in the population.",
          "max_generations": "The maximum number of generations to run the algorithm."
        }
      },
      {
        "algorithm": "Simulated Annealing",
        "solution_type": "Approximate",
        "description": "Simulates the physical annealing process to gradually find the optimal solution, suitable for large-scale problems.",
        "suitable_graph_size": "Suitable for medium to large graphs (up to 50-100 nodes), providing a good approximation to the optimal solution.",
        "time_complexity": "O(n^2 * log(n))",
        "input": {
          "adjacency_list": "A complete weighted graph represented as an adjacency matrix or edge list.",
          "start_node": "The starting node for the traveling salesman problem.",
          "initial_temperature": "The starting temperature for the annealing process.",
          "cooling_rate": "The rate at which the temperature decreases.",
          "max_iterations": "The maximum number of iterations to run the algorithm."
        }
      }
    ],
    "GraphColoring": [
      {
        "algorithm": "Backtracking",
        "solution_type": "Optimal",
        "description": "Uses depth-first search and backtracking to find the minimal coloring number.",
        "suitable_graph_size": "Suitable for small graphs with fewer than 20 nodes. For larger graphs, the time complexity becomes prohibitive.",
        "time_complexity": "O(2^n)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      },
      {
        "algorithm": "Greedy Coloring Algorithm",
        "solution_type": "Approximate",
        "description": "Colors each node based on its order, trying to minimize the number of colors, but it may not give the optimal solution.",
        "suitable_graph_size": "Suitable for medium to small graphs, especially for larger graphs where the greedy approach can provide a good approximation quickly.",
        "time_complexity": "O(n^2)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      },
      {
        "algorithm": "DSATUR Algorithm",
        "solution_type": "Approximate",
        "description": "Based on the degree of the node and the number of colored adjacent nodes, it selects the next node to color, typically providing a good approximation.",
        "suitable_graph_size": "Suitable for medium to large graphs, with up to hundreds of nodes, providing a good approximation in reasonable time.",
        "time_complexity": "O(n^2)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      },
      {
        "algorithm": "Minimum Coloring Algorithm",
        "solution_type": "Optimal",
        "description": "Finds the minimum coloring number of a graph using exact coloring methods, often relying on dynamic programming or exhaustive search.",
        "suitable_graph_size": "Suitable for small graphs (less than 50 nodes). Due to its computational complexity, it is ideal for finding the optimal solution in small graphs.",
        "time_complexity": "O(2^n)",
        "input": {
          "adjacency_list": "An undirected graph represented as an adjacency matrix or adjacency list."
        }
      }
    ],
    "ShortestPath": [
      {
        "algorithm": "Dijkstra's Algorithm",
        "solution_type": "Optimal",
        "description": "Finds the shortest path from a single source node to all other nodes in a graph with non-negative weights.",
        "suitable_graph_size": "Suitable for graphs with up to tens of thousands of nodes. The algorithm is efficient, especially for sparse graphs.",
        "time_complexity": "O(V^2) or O(E + V log V) with a priority queue",
        "input": {
          "adjacency_list": "A weighted graph represented as an adjacency matrix or adjacency list.",
          "source": "The source node for the shortest path.",
          "goal": "The destination node for the shortest path.",
        }
      },
      {
        "algorithm": "Bellman-Ford Algorithm",
        "solution_type": "Optimal",
        "description": "Handles graphs with negative weights and detects negative weight cycles.",
        "suitable_graph_size": "Suitable for small to medium-sized graphs (up to 10,000 nodes). It's less efficient than Dijkstra's for graphs without negative weights.",
        "time_complexity": "O(V * E)",
        "input": {
          "adjacency_list": "A weighted graph represented as an adjacency matrix or adjacency list.",
          "source": "The source node for the shortest path.",
          "goal": "The destination node for the shortest path.",
        }
      },
      {
        "algorithm": "Floyd-Warshall Algorithm",
        "solution_type": "Optimal",
        "description": "Finds the shortest paths between all pairs of nodes in a graph.",
        "suitable_graph_size": "Suitable for small to medium graphs. The algorithm is not efficient for large graphs due to its cubic time complexity.",
        "time_complexity": "O(V^3)",
        "input": {
          "adjacency_list": "A weighted graph represented as an adjacency matrix."
        }
      },
      {
        "algorithm": "A* Search Algorithm",
        "solution_type": "Approximate",
        "description": "Uses heuristics to find the shortest path between nodes more efficiently than Dijkstra's algorithm in specific cases.",
        "suitable_graph_size": "Suitable for medium-sized graphs (up to tens of thousands of nodes) and often used in pathfinding in game development.",
        "time_complexity": "O(E + V log V)",
        "input": {
          "adjacency_list" "A weighted graph represented as an adjacency matrix or adjacency list.",
          "source": "The source node for the shortest path.",
          "goal": "The destination node for the shortest path.",
          "heuristic": "A heuristic function to guide the search towards the goal."
        }
      }
    ],
    "HamiltonianPath": [
      {
        "algorithm": "Backtracking",
        "solution_type": "Optimal",
        "description": "Explores all possible paths to find a Hamiltonian path, ensuring that each vertex is visited exactly once.",
        "suitable_graph_size": "Suitable for small graphs (up to 15 nodes), as the time complexity grows exponentially.",
        "time_complexity": "O(2^n)",
        "inputs": {
          "adjacency_list": "A list of lists or adjacency matrix representing the graph's edges."
        }
      },
      {
        "algorithm": "Dynamic Programming (Held-Karp variant)",
        "solution_type": "Optimal",
        "description": "Uses dynamic programming to reduce the complexity of solving the Hamiltonian path problem, though still exponential in nature.",
        "suitable_graph_size": "Suitable for small to medium-sized graphs (up to 20 nodes). It's an improvement over brute-force backtracking.",
        "time_complexity": "O(n^2 * 2^n)",
        "inputs": {
          "adjacency_list": "A list of lists or adjacency matrix representing the graph's edges."
        }
      },
      {
        "algorithm": "Approximation (Greedy Approach)",
        "solution_type": "Approximate",
        "description": "Uses a greedy approach to construct a Hamiltonian path by picking the next unvisited vertex closest to the current vertex, though it may not always find a solution.",
        "suitable_graph_size": "Suitable for large graphs where finding the exact solution is infeasible. Greedy algorithms provide quick solutions but not necessarily valid Hamiltonian paths.",
        "time_complexity": "O(n^2)",
        "inputs": {
          "adjacency_list": "A list of lists or adjacency matrix representing the graph's edges."
        }
      }
    ]
  }
}    
"""