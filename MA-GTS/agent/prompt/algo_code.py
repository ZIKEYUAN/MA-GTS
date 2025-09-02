import sys
import time
from itertools import combinations
import numpy as np

import heapq

def transform_dict(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        new_list = []
        for item in value:
            for sub_key, sub_value in item.items():
                new_list.append((int(sub_key), int(sub_value)))
        output_dict[int(key)] = new_list
    return output_dict
def transform_list(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        new_list = []
        for item in value:
            print(item)
            if item == None:
                new_list = []
            else:
                new_list.append((int(item[0]), int(item[1])))
        output_dict[int(key)] = new_list
    return output_dict   

def tsp_dynamic_programming(adjacency_list):
    r"""Solves the Traveling Salesman Problem (TSP) using Dynamic Programming.

    Args:
        adjacency_list (dict): The adjacency list representing the graph. 

    Returns:
        tuple: A tuple containing:
            - A list of nodes (in terms of node IDs) representing the TSP tour.
            - The minimum tour cost.
            - The time taken to compute the solution.
    """
    
    
    # Ensure all keys are int
    adjacency_list = {int(k): v for k,v in adjacency_list.items()}
    if isinstance(adjacency_list[0][0],dict):
        adjacency_list = transform_dict(adjacency_list)
    start_node = 0
    # Collect all nodes from keys and their neighbors
    node_set = set(adjacency_list.keys())
    for node, neighbors in adjacency_list.items():
        for (neighbor, dist) in neighbors:
            node_set.add(neighbor)

    # Create the node list from the set
    nodes = sorted(list(node_set))
    n = len(nodes)

    # Convert start_node to start_idx
    if start_node not in nodes:
        raise ValueError(f"Start node {start_node} not found in graph nodes {nodes}.")
    start_idx = nodes.index(start_node)

    # Initialize the distance matrix with sys.maxsize
    distance_matrix = [[sys.maxsize] * n for _ in range(n)]

    # Fill the distance matrix
    for node, neighbors in adjacency_list.items():
        node_idx = nodes.index(node)
        for neighbor, distance in neighbors:
            neighbor_idx = nodes.index(neighbor)
            distance_matrix[node_idx][neighbor_idx] = distance
            distance_matrix[neighbor_idx][node_idx] = distance

    start_time = time.time()
    all_visited = (1 << n) - 1
    memo = {}

    def dp(current, visited):
        # current和visited都是基于索引的
        if (current, visited) in memo:
            return memo[(current, visited)]

        if visited == all_visited:
            # 所有节点都访问过，返回至起点(用start_idx访问distance_matrix)
            return distance_matrix[current][start_idx], [current, start_idx]

        min_cost = sys.maxsize
        min_path = []

        for next_node in range(n):
            if (visited >> next_node) & 1 == 0:
                cost, path = dp(next_node, visited | (1 << next_node))
                cost += distance_matrix[current][next_node]
                if cost < min_cost:
                    min_cost = cost
                    min_path = [current] + path

        memo[(current, visited)] = (min_cost, min_path)
        return memo[(current, visited)]

    cost, path = dp(start_idx, 1 << start_idx)
    end_time = time.time()

    # 将路径中的索引映射回实际的节点ID
    path = [nodes[i] for i in path]
    return (path, cost, end_time - start_time)

def tsp_greedy_nearest_neighbor(adjacency_list):
    r"""Solves the Traveling Salesman Problem (TSP) using Dynamic Programming.

    Args:
        adjacency_list (dict): The adjacency list representing the graph.

    Returns:
        tuple: A tuple containing:
            - A list of nodes (in terms of node IDs) representing the TSP tour.
            - The minimum tour cost.
            - The time taken to compute the solution.
    """
    # Extract adjacency list and start node
    # print("==input_data==")
    # print(input_data)
    # adjacency_list = input_data["adjacency_list"]
    # start_node = input_data["start_node"]
    
    # Ensure all node keys and neighbors are integers
    start_node=0
    adjacency_list = {int(k): v for k,v in adjacency_list.items()}
    if isinstance(adjacency_list[0][0],dict):
        adjacency_list = transform_dict(adjacency_list)
    
    # Collect all nodes from keys and neighbors to ensure no missing nodes
    node_set = set(adjacency_list.keys())
    for node, neighbors in adjacency_list.items():
        for (neighbor, dist) in neighbors:
            node_set.add(neighbor)
    nodes = sorted(list(node_set))
    n = len(nodes)
    
    # Check if start_node is in the node list
    if start_node not in nodes:
        raise ValueError(f"Start node {start_node} not found among nodes {nodes}.")

    start_idx = nodes.index(start_node)

    # Build a distance matrix for quick look-ups
    distance_matrix = [[sys.maxsize] * n for _ in range(n)]
    for node, neighbors in adjacency_list.items():
        u = nodes.index(node)
        for (neighbor, dist) in neighbors:
            v = nodes.index(neighbor)
            distance_matrix[u][v] = dist
            distance_matrix[v][u] = dist

    # Start the timer
    start_time = time.time()

    # Nearest-neighbor heuristic
    visited = [False] * n
    visited[start_idx] = True
    path_idx = [start_idx]  # store indices first, convert later
    current_idx = start_idx

    for _ in range(n - 1):
        next_idx = None
        min_dist = sys.maxsize
        # Find the closest unvisited node
        for j in range(n):
            if not visited[j] and distance_matrix[current_idx][j] < min_dist:
                min_dist = distance_matrix[current_idx][j]
                next_idx = j
        
        if next_idx is None:
            # No unvisited nodes found, graph might be incomplete or disconnected
            raise ValueError("No unvisited nodes found. The graph might be disconnected.")
        
        visited[next_idx] = True
        path_idx.append(next_idx)
        current_idx = next_idx
    
    # Return to the start node
    path_idx.append(start_idx)

    # Calculate total length
    total_length = 0
    for i in range(len(path_idx) - 1):
        total_length += distance_matrix[path_idx[i]][path_idx[i+1]]

    end_time = time.time()

    # Convert indices back to node IDs
    path = [nodes[i] for i in path_idx]

    return (path, total_length, end_time - start_time)

def graph_coloring_backtracking(adjacency_list):
    r"""Solves the Graph Coloring problem using backtracking.

    Args:
        adjacency_list (dict): The adjacency list representing the graph. 

    Returns:
        tuple: A tuple containing:
            - An integer representing the minimum number of colors required.
            - A list of integers representing the color assigned to each node.
            - The time taken to compute the solution.
    """
    import time

    start_time = time.time()

    # Extract nodes from adjacency list
    nodes = list(adjacency_list.keys())
    nodes = [int(i) for i in nodes]
    n = len(nodes)
    color = [0] * n  # Initialize all nodes as uncolored

    # Create adjacency matrix from adjacency list
    adjacency_matrix = [[0] * n for _ in range(n)]
    for node, neighbors in adjacency_list.items():
        node = int(node)
        for neighbor, _ in neighbors:
            neighbor = int(neighbor)
            adjacency_matrix[node][neighbor] = 1
            adjacency_matrix[neighbor][node] = 1

    def is_valid(node, c):
        """Checks if the color assignment is valid for the given node."""
        for neighbor in range(len(adjacency_matrix)):
            if adjacency_matrix[node][neighbor] == 1 and color[neighbor] == c:
                return False
        return True

    def backtrack(node, m):
        """Backtracking function to assign colors to nodes."""
        if node == n:  # All nodes are colored
            return True

        for c in range(1, m + 1):  # Try each color
            if is_valid(node, c):
                color[node] = c  # Assign color
                if backtrack(node + 1, m):  # Recurse for the next node
                    return True
                color[node] = 0  # Undo assignment

        return False

    # Start with the smallest number of colors and find the solution
    for m in range(1, n + 1):
        if backtrack(0, m):
            end_time = time.time()
            return m, color, end_time - start_time

    end_time = time.time()
    return n, color, end_time - start_time  # In the worst case, return n colorsme  # In the worst case, return n colors

def graph_coloring_greedy(adjacency_list):
    r"""Solves the Graph Coloring problem using greedy.

    Args:
        adjacency_list (dict): The adjacency list representing the graph. 

    Returns:
        tuple: A tuple containing:
            - An integer representing the minimum number of colors required.
            - A list of integers representing the color assigned to each node.
            - The time taken to compute the solution.
    """
    start_time = time.time()
    num_nodes = len(adjacency_list)

    # 构造邻接矩阵
    graph = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, neighbors in adjacency_list.items():
        node_idx = int(node)
        for neighbor, weight in neighbors:
            neighbor = int(neighbor)
            graph[node_idx][neighbor] = 1  # 由于是无权图，这里只需要标记 1
            graph[neighbor][node_idx] = 1

# 贪心算法求解图着色问题
    n = len(graph)  # 节点数
    color = [-1] * n  # 初始化所有节点为未着色 (-1)

    # 遍历所有节点，按顺序为每个节点分配颜色
    for node in range(n):
        # 找出当前节点的所有邻居的颜色
        neighbor_colors = set()
        for neighbor in range(n):
            if graph[node][neighbor] == 1 and color[neighbor] != -1:  # 相邻节点已着色
                neighbor_colors.add(color[neighbor])

        # 找到第一个未被邻居占用的颜色
        chosen_color = 1
        while chosen_color in neighbor_colors:
            chosen_color += 1

        color[node] = chosen_color  # 为当前节点分配颜色

    # 最小着色数为分配的颜色数的最大值
    min_colors = max(color)
    end_time = time.time()
    return min_colors, color, end_time - start_time

def vertex_cover_brute_force(adjacency_list):
    r"""Solves the vertex cover problem using brute force.

    Args:
        adjacency_list (dict): The adjacency list representing the graph. 

    Returns:
        tuple: A tuple containing:
            - The list of vertices in the minimum vertex cover.
            - The time taken to compute the solution.
    """
    start_time = time.time()

    # Step 1: Preprocess the adjacency list (convert keys and values to integers)
    processed_adjacency_list = {}
    for key, neighbors in adjacency_list.items():
        int_key = int(key)  # Convert the key to an integer
        int_neighbors = [(int(neighbor), weight) for neighbor, weight in neighbors]  # Convert neighbors to integers
        processed_adjacency_list[int_key] = int_neighbors

    # Number of vertices in the graph
    n = len(processed_adjacency_list)

    # Step 2: Define a helper function to check if a set is a valid vertex cover
    def is_vertex_cover(vertex_set):
        """
        Checks if the given vertex_set is a valid vertex cover.
        :param vertex_set: set
            A set of vertices to be checked.
        :return: bool
            True if the vertex_set is a valid vertex cover, False otherwise.
        """
        for u, neighbors in processed_adjacency_list.items():
            for v, _ in neighbors:  # Ignore edge weights
                # If edge (u, v) is not covered by vertex_set
                if u not in vertex_set and v not in vertex_set:
                    return False
        return True

    # Step 3: Brute force enumeration of subsets of vertices
    for k in range(1, n + 1):
        for vertex_set in combinations(range(n), k):
            if is_vertex_cover(set(vertex_set)):  # Convert tuple to set
                end_time = time.time()
                cost_time = end_time - start_time
                return list(vertex_set), len(vertex_set),cost_time

    # If no vertex cover is found (should not happen for valid inputs)
    end_time = time.time()
    return [], 0,end_time - start_time

def dijkstra_shortest_path(adjacency_list, start_node, end_node):
    r"""Solves the Single Source Shortest Path problem using Dijkstra's algorithm on an undirected graph.

    Args:
        adjacency_list (dict): The adjacency list representing the graph.
        start_node (int): The source node.
        end_node (int): The destination node.

    Returns:
        tuple: A tuple containing:
            - An integer representing the total cost of the shortest path.
            - A list of nodes representing the shortest path from start to end.
            - The time taken to compute the solution.
    """
    start_time = time.time()
    start_node = int(start_node)
    end_node = int(end_node)
    adjacency_list = {int(k): v for k,v in adjacency_list.items()}
    if isinstance(adjacency_list[0][0],dict):
        adjacency_list = transform_dict(adjacency_list)
    elif isinstance(adjacency_list[0][0],list):
        adjacency_list = transform_list(adjacency_list)
    print(adjacency_list)
    # Step 1: 处理为对称的无向图邻接表
    undirected_adj = {}
    for node, neighbors in adjacency_list.items():
        undirected_adj.setdefault(node, [])
        for neighbor, weight in neighbors:
            undirected_adj[node].append((neighbor, weight))
            undirected_adj.setdefault(neighbor, [])
            # 避免重复插入（只添加对称边 if 不存在）
            if node not in [n for n, _ in undirected_adj[neighbor]]:
                undirected_adj[neighbor].append((node, weight))

    # Step 2: 初始化 Dijkstra
    num_nodes = len(undirected_adj)
    distances = {node: float('inf') for node in undirected_adj}
    predecessors = {node: None for node in undirected_adj}
    distances[start_node] = 0

    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end_node:
            break

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in undirected_adj.get(current_node, []):
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Step 3: 回溯路径
    path = []
    current = end_node
    if distances[end_node] != float('inf'):
        while current is not None:
            path.insert(0, current)
            current = predecessors[current]
    else:
        path = None

    end_time = time.time()
    return distances[end_node], path

# adjacency_list = {'0': [['1', 4], ['3', 4], ['2', 3], ['4', 1]], '1': [['3', 3], ['4', 1]], '2': [['3', 3], ['4', 2]], '3': [['4', 1]], '4': []}
# adjacency_list_2 = {
#     0: [(3, 1), (2, 1)],
#     1: [(3, 3), (4, 4)],
#     2: [(4, 3)],
#     3: [(4, 4)],
#     4: []
#   }
# print(dijkstra_shortest_path(adjacency_list_2,1,2))