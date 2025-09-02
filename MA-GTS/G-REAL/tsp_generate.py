import networkx as nx
import random
import numpy as np
import itertools
import json
from tqdm import tqdm
import sys
import time   
import heapq
from pathlib import Path
import math
import multiprocessing as mp
from functools import partial
import numpy as np
from numba import njit, prange
SAVE_DIR = "../data/TSP_data_50/"
ALGORITHM_LIST = [
    # "brute_force",
    # "tsp_brute_force_parallel",
    "dynamic_programming",
    # "tsp_dynamic_programming_numba_parallel",
    # "branch_and_bound",
    "christofides_algorithm",
    "greedy_algorithm",
    "nearest_neighbor_algorithm",
    "simulated_annealing",
    "genetic_algorithm"]
ALG_DIC = {
    "brute_force"
} 
PLACE = {
    "Amber Plaza": "A bustling central square surrounded by cafes, boutiques, and street performers.",
    "Beacon Tower": "The tallest building in the city, offering panoramic views and a rotating rooftop restaurant.",
    "Cobalt Market": "A vibrant marketplace where merchants sell exotic goods and fresh produce from all over.",
    "Duskwood Park": "A sprawling urban park filled with dense trees, walking trails, and a serene lake.",
    "Echo Station": "The city’s largest transportation hub, always alive with the sound of trains and announcements.",
    "Flare Alley": "A narrow, colorful street lined with neon-lit bars and underground clubs.",
    "Gilded Archway": "A historic landmark leading to the city’s oldest district, adorned with intricate carvings.",
    "Haven Docks": "The city’s bustling port area, filled with cargo ships, seafood stalls, and lively taverns.",
    "Ironbridge Crossing": "A massive steel bridge connecting the industrial zone with the city center.",
    "Jade Fountain": "A tranquil plaza centered around a beautiful fountain made of green stone.",
    "King’s Row": "A luxurious shopping street lined with high-end stores and designer boutiques.",
    "Lighthouse Point": "A scenic overlook by the bay with a historic lighthouse and picnic spots.",
    "Moonlit Promenade": "A romantic walkway along the riverbank, lit by soft lanterns at night.",
    "Nimbus Plaza": "A futuristic square surrounded by glass skyscrapers and interactive digital art installations.",
    "Oakshade Library": "The city’s largest library, featuring towering bookshelves and cozy reading nooks.",
    "Pennywhistle Arcade": "A vintage entertainment district with old-style theaters, arcades, and street performers.",
    "Quartz District": "A modern financial center with sleek skyscrapers and luxury dining establishments.",
    "Riverstone Wharf": "A bustling area along the river, known for its floating markets and seafood restaurants.",
    "Skyline Gardens": "A rooftop garden atop the central mall, offering breathtaking views and quiet retreats.",
    "Temple Square": "A historic site featuring a grand temple surrounded by artisan shops and open courtyards.",
    "Umbra Theater": "A grand opera house with velvet interiors and a reputation for world-class performances.",
    "Velvet Corner": "A hidden alley filled with cozy cafes, second-hand bookstores, and intimate music venues.",
    "Westgate Station": "A major train terminal, connecting the city to nearby towns and regions.",
    "Yarrow Plaza": "A cultural hub with art galleries, street food stalls, and live performances every evening.",
    "Zenith Arena": "A state-of-the-art stadium for concerts, sports events, and major public gatherings.",
    "Azure Gardens": "An expansive botanical garden featuring rare plants, greenhouses, and water features.",
    "Brass Lantern Tavern": "A cozy pub famous for its handcrafted beers and warm, rustic interiors.",
    "Copper Clock Square": "A historic plaza with a towering clock surrounded by antique shops and cafes.",
    "Dragon’s Gate": "An iconic stone gate marking the entrance to the vibrant Chinatown district.",
    "Evergreen Circle": "A quiet residential park with a children’s playground and small community events.",
    "Flint Forge Quarter": "An old industrial area now revitalized with art studios and trendy cafes.",
    "Granite Plaza": "A corporate plaza with water fountains and benches, popular among office workers during lunch.",
    "Horizon Mall": "A massive indoor shopping center with everything from luxury brands to casual dining.",
    "Ivory Spire Cathedral": "A towering cathedral with stunning stained-glass windows and a melodic bell tower.",
    "Jasper Marina": "A sleek marina hosting luxury yachts, seafood dining, and weekend sailing lessons.",
    "Knight’s Market": "A medieval-themed marketplace with handcrafted goods and costumed vendors.",
    "Lunar Pier": "A picturesque wooden pier with food stalls, fishing spots, and a small amusement park.",
    "Mosaic Plaza": "A public square adorned with colorful tile art and surrounded by artisan coffee shops.",
    "Northwind Tower": "A modern skyscraper with a rotating observation deck and sky-high dining.",
    "Opal Theater": "An avant-garde cinema showing independent films and hosting film festivals.",
    "Primrose Boulevard": "A tree-lined street with boutique stores, local bakeries, and street performers.",
    "Quarry Point": "An old quarry turned into a unique rock-climbing park and open-air amphitheater.",
    "Rosewood Hall": "A grand event hall hosting galas, weddings, and large conferences.",
    "Silvercrest Observatory": "A high-tech observatory where visitors can stargaze and learn about astronomy.",
    "Twilight Harbor": "A picturesque dock area with twinkling lights and waterfront dining.",
    "Union Square Market": "A farmers’ market offering fresh produce, handmade crafts, and live music.",
    "Maplewood Conservatory": "A sprawling indoor botanical garden with exotic plants and tranquil waterfalls.",
    "Sapphire Arena": "A massive sports and concert venue surrounded by restaurants and fan zones.",
    "Shadowbridge Arcade": "A covered walkway filled with specialty stores, cafes, and hidden speakeasies.",
    "Willowshade Pavilion": "A cultural pavilion hosting open-air art exhibits, food festivals, and community events."
}
#===============精确算法=====================
def tsp_brute_force(distance_matrix, start_node):
    start_time = time.time()
    n = len(distance_matrix)
    nodes = list(range(n))
    nodes.remove(start_node)  # 除去起始节点
    min_path = []
    min_cost = sys.maxsize

    # 生成所有可能的城市排列
    for perm in itertools.permutations(nodes):
        current_cost = 0
        current_path = [start_node] + list(perm) + [start_node]
        # 计算当前路径的总长度
        for i in range(len(current_path) - 1):
            current_cost += distance_matrix[current_path[i]][current_path[i + 1]]
        # 更新最短路径和长度
        if current_cost < min_cost:
            min_cost = current_cost
            min_path = current_path
    end_time = time.time()
    return min_path, min_cost,end_time - start_time

def tsp_dynamic_programming(distance_matrix, start_node):
    start_time = time.time()
    n = len(distance_matrix)
    all_visited = (1 << n) - 1  # 所有节点都被访问的位掩码
    memo = {}

    def dp(current, visited):
        if (current, visited) in memo:
            return memo[(current, visited)]
        if visited == all_visited:
            return distance_matrix[current][start_node], [current, start_node]
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

    cost, path = dp(start_node, 1 << start_node)
    end_time = time.time()
    return path, cost , end_time - start_time

def tsp_branch_and_bound(distance_matrix, start_node):
    start_time = time.time()
    n = len(distance_matrix)
    min_cost = sys.maxsize
    best_path = []
    priority_queue = []

    # 计算下界的函数（简单估计）
    def calculate_lower_bound(visited):
        bound = 0
        unvisited = [i for i in range(n) if i not in visited]
        # 加上当前路径的长度
        for i in range(len(visited) - 1):
            bound += distance_matrix[visited[i]][visited[i + 1]]
        # 加上从当前节点到未访问节点的最小边
        if unvisited:
            last = visited[-1]
            min_edge = min([distance_matrix[last][i] for i in unvisited])
            bound += min_edge
            # 加上未访问节点的最小出边
            for i in unvisited:
                min_edge = min([distance_matrix[i][j] for j in range(n) if j != i])
                bound += min_edge
            # 加上未访问节点回到起点的最小边
            min_edge = min([distance_matrix[i][start_node] for i in unvisited])
            bound += min_edge
        else:
            # 所有节点已访问，回到起始节点
            bound += distance_matrix[visited[-1]][start_node]
        return bound

    # 初始节点
    heapq.heappush(priority_queue, (0, [start_node]))
    while priority_queue:
        cost, path = heapq.heappop(priority_queue)
        if len(path) == n + 1:
            if cost < min_cost:
                min_cost = cost
                best_path = path
            continue
        if cost >= min_cost:
            continue
        last = path[-1]
        for i in range(n):
            if i not in path:
                new_cost = cost + distance_matrix[last][i]
                new_path = path + [i]
                lower_bound = calculate_lower_bound(new_path)
                if lower_bound < min_cost:
                    heapq.heappush(priority_queue, (new_cost, new_path))
        # 当所有节点都已访问，回到起始节点
        if len(path) == n:
            new_cost = cost + distance_matrix[last][start_node]
            new_path = path + [start_node]
            if new_cost < min_cost:
                min_cost = new_cost
                best_path = new_path
    end_time = time.time()
    return best_path, min_cost , end_time - start_time
#===============近似算法=====================
def christofides_algorithm(adj_matrix, start_node):
    start_time = time.time()
    n = len(adj_matrix)
    G = nx.Graph()

    # 将邻接矩阵中的边添加到图中
    for i in range(n):
        for j in range(i+1, n):
            weight = adj_matrix[i][j]
            G.add_edge(i, j, weight=weight)

    # 计算最小生成树
    mst = nx.minimum_spanning_tree(G, weight='weight')

    # 找到最小生成树中度数为奇数的节点
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]

    # 从原始图中提取这些奇度节点构成的子图
    subgraph = G.subgraph(odd_degree_nodes)

    # 计算子图的最小权重完美匹配
    mwpm = nx.algorithms.matching.min_weight_matching(subgraph, weight='weight')

    # 合并最小生成树和匹配结果，形成多重图
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(mwpm)

    # 在多重图中寻找欧拉回路
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=start_node))

    # 将欧拉回路转换为哈密顿回路，跳过重复访问的节点
    path = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)
        if v not in visited:
            path.append(v)
            visited.add(v)

    # 确保回到起始节点
    if path[0] != path[-1]:
        path.append(path[0])

    # 计算总路径长度
    total_length = sum(adj_matrix[path[i]][path[i+1]] for i in range(len(path) - 1))
    end_time = time.time()

    return path, total_length, end_time - start_time


def greedy_algorithm(adj_matrix, start_node):
    start_time = time.time()
    n = len(adj_matrix)
    visited = [False] * n
    path = [start_node]
    visited[start_node] = True
    total_length = 0
    
    current_node = start_node
    
    # 进行n-1次选择，选择最近的未访问节点
    for _ in range(n - 1):
        min_distance = float('inf')
        next_node = None
        
        # 遍历所有节点，选择距离当前节点最近的未访问节点
        for neighbor in range(n):
            if not visited[neighbor] and adj_matrix[current_node][neighbor] < min_distance:
                min_distance = adj_matrix[current_node][neighbor]
                next_node = neighbor
        
        # 将选择的节点加入路径，并更新当前节点
        path.append(next_node)
        visited[next_node] = True
        total_length += min_distance
        current_node = next_node
    
    # 回到起始节点，完成闭环
    total_length += adj_matrix[current_node][start_node]
    path.append(start_node)
    
    end_time = time.time()
    
    return path, total_length, end_time - start_time

def nearest_neighbor_algorithm(adj_matrix, start_node):
    start_time = time.time()
    n = len(adj_matrix)
    visited = [False] * n
    path = [start_node]
    visited[start_node] = True
    current_node = start_node

    while len(path) < n:
        next_node = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and adj_matrix[current_node][i] < min_dist:
                min_dist = adj_matrix[current_node][i]
                next_node = i
        path.append(next_node)
        visited[next_node] = True
        current_node = next_node

    # 回到起始节点
    path.append(start_node)

    # 计算总长度
    total_length = 0
    for i in range(len(path) - 1):
        total_length += adj_matrix[path[i]][path[i+1]]
    end_time = time.time()
    return path, total_length,end_time - start_time

#==================启发式-模拟退火==========================
def calculate_cost(adj_matrix, solution):
    cost = 0
    for i in range(len(solution) - 1):
        cost += adj_matrix[solution[i]][solution[i + 1]]
    return cost
def simulated_annealing(adj_matrix, start_node):
    start_time = time.time()
    num_cities = len(adj_matrix)
    current_solution = list(range(num_cities))
    current_solution.remove(start_node)
    random.shuffle(current_solution)
    current_solution = [start_node] + current_solution + [start_node]
    best_solution = current_solution.copy()
    best_cost = calculate_cost(adj_matrix, best_solution)
    current_temp = 10000
    cooling_rate = 0.995
    absolute_temp = 0.00001

    while current_temp > absolute_temp:
        new_solution = current_solution.copy()
        # 随机交换两个城市
        idx1 = random.randint(1, num_cities - 1)
        idx2 = random.randint(1, num_cities - 1)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        new_cost = calculate_cost(adj_matrix, new_solution)

        cost_diff = new_cost - best_cost
        if cost_diff < 0 or math.exp(-cost_diff / current_temp) > random.random():
            current_solution = new_solution.copy()
            best_cost = new_cost
            best_solution = new_solution.copy()

        current_temp *= cooling_rate
    end_time = time.time()
    return best_solution, best_cost, end_time - start_time

#==================启发式-遗传算法==========================
def genetic_algorithm(adj_matrix, start_node, population_size=100, generations=500, mutation_rate=0.01):
    start_time = time.time()
    num_cities = len(adj_matrix)
    population = generate_initial_population(population_size, num_cities, start_node)
    for _ in range(generations):
        fitness_scores = [1 / calculate_cost(adj_matrix, individual) for individual in population]
        new_population = []
        for _ in range(population_size):
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            child = crossover(parent1, parent2, start_node)
            child = mutate(child, mutation_rate, start_node)
            new_population.append(child)
        population = new_population

    best_individual = min(population, key=lambda x: calculate_cost(adj_matrix, x))
    best_cost = calculate_cost(adj_matrix, best_individual)
    end_time = time.time()
    return best_individual, best_cost, end_time - start_time

def generate_initial_population(size, num_cities, start_node):
    population = []
    for _ in range(size):
        individual = list(range(num_cities))
        individual.remove(start_node)
        random.shuffle(individual)
        individual = [start_node] + individual + [start_node]
        population.append(individual)
    return population

def calculate_cost(adj_matrix, solution):
    cost = 0
    for i in range(len(solution) - 1):
        cost += adj_matrix[solution[i]][solution[i + 1]]
    return cost

def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current > pick:
            return individual

def crossover(parent1, parent2, start_node):
    child = [start_node] + [None] * (len(parent1) - 2) + [start_node]
    start_pos = random.randint(1, len(parent1) - 3)
    end_pos = random.randint(start_pos, len(parent1) - 3)
    child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1]
    ptr = 1
    for gene in parent2:
        if gene not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = gene
    return child

def mutate(individual, mutation_rate, start_node):
    for swapped in range(1, len(individual) - 1):
        if random.random() < mutation_rate:
            swap_with = random.randint(1, len(individual) - 2)
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

#===========多线程暴力解=================
def compute_cost(distance_matrix, start_node, perm):
    current_cost = 0
    current_path = [start_node] + list(perm) + [start_node]
    # Calculate total length of current path
    for i in range(len(current_path) - 1):
        current_cost += distance_matrix[current_path[i]][current_path[i + 1]]
    return current_cost, current_path

def tsp_brute_force_parallel(distance_matrix, start_node):
    start_time = time.time()
    n = len(distance_matrix)
    nodes = list(range(n))
    nodes.remove(start_node)  # Remove starting node
    min_path = []
    min_cost = sys.maxsize

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    # Create a partial function to fix distance_matrix and start_node
    compute_partial = partial(compute_cost, distance_matrix, start_node)

    # Generate permutations without storing them all in memory
    permutations = itertools.permutations(nodes)

    # Process permutations in parallel
    # Adjust chunksize to balance between memory usage and performance
    chunksize = 1000
    results = pool.imap_unordered(compute_partial, permutations, chunksize=chunksize)

    try:
        for current_cost, current_path in results:
            if current_cost < min_cost:
                min_cost = current_cost
                min_path = current_path
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    end_time = time.time()
    return min_path, min_cost, end_time - start_time

@njit
def tsp_dp_numba(distance_matrix, start_node):
    n = len(distance_matrix)
    all_visited = (1 << n) - 1  # All nodes visited bitmask

    dp_cost = np.full((1 << n, n), sys.maxsize, dtype=np.int64)
    dp_prev = np.full((1 << n, n), -1, dtype=np.int64)
    dp_cost[1 << start_node][start_node] = 0  # Cost to reach start_node is 0

    for visited in range(1 << n):
        for last in range(n):
            if (visited >> last) & 1 == 0:
                continue
            cost_so_far = dp_cost[visited][last]
            for next_node in range(n):
                if (visited >> next_node) & 1 == 0:
                    next_visited = visited | (1 << next_node)
                    cost = cost_so_far + distance_matrix[last][next_node]
                    if cost < dp_cost[next_visited][next_node]:
                        dp_cost[next_visited][next_node] = cost
                        dp_prev[next_visited][next_node] = last

    # After filling dp table, find the minimum cost to return to start_node
    min_cost = sys.maxsize
    last_node = -1
    for last in range(n):
        if last == start_node:
            continue
        cost_so_far = dp_cost[all_visited][last]
        if cost_so_far == sys.maxsize:
            continue  # Skip if this state was never reached
        cost = cost_so_far + distance_matrix[last][start_node]
        if cost < min_cost:
            min_cost = cost
            last_node = last

    # Reconstruct path
    path = [start_node]
    visited = all_visited
    current_node = last_node
    while current_node != start_node:
        path.append(current_node)
        prev_node = dp_prev[visited][current_node]
        visited = visited & ~(1 << current_node)
        current_node = prev_node
    path.append(start_node)  # Return to start node
    path.reverse()
    return path, min_cost

def tsp_dynamic_programming_parallel(distance_matrix, start_node):
    start_time = time.time()
    distance_matrix = np.array(distance_matrix)
    path, min_cost = tsp_dp_numba(distance_matrix, start_node)
    end_time = time.time()
    return path, min_cost, end_time - start_time

def generate_complete_graph(n):
    G = nx.complete_graph(n)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)

    return G

def print_graph(G):
    for (u, v, wt) in G.edges(data='weight'):
        print(f"从节点 {u} 到节点 {v} 的边，权重为 {wt:.2f}")

def get_adjacency_matrix(G):
    n = G.number_of_nodes()
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for (u, v, wt) in G.edges(data='weight'):
        adjacency_matrix[u][v] = wt
        adjacency_matrix[v][u] = wt  # 因为是无向图，矩阵是对称的
    return adjacency_matrix

def get_random_numbers(n):
    """
    从0到49之间随机取n个不重复的数。
    :param n: 要取的数字个数，必须小于等于50
    :return: 一个包含n个随机数的列表
    """
    if n > 50 or n < 0:
        raise ValueError("n must >=0 and <50")
    return random.sample(range(50), n)

def get_start_q(n,place_list,G):
    q = "Our company handles deliveries across a busy urban area, and today we have {n} distinct delivery points to cover. The delivery driver will start from our central warehouse and needs to drop off packages at each location before returning to the warehouse. Since these delivery points are scattered throughout different parts of the city, we’re looking to find the most efficient route to minimize the total distance traveled. This will help us save on fuel, reduce delivery times, and improve our overall efficiency.\nThe warehouse, is located near the city center. Each location represents a different type of business or residential area with unique delivery requirements:\n".format(n=n)
    index_list = get_random_numbers(n)
    items = list(PLACE.items())
    name_list = ["Warehouse"] + [items[i][0] for i in index_list]
    all = ""
    for i in index_list:
        location = items[i]
        name,discribe = location
        location_discribe = name + ": " + discribe + "\n"
        all = all + location_discribe
    q = q + all + "\n" + "Each pair of points has a different travel distance between them, based on city traffic patterns and street layouts. Here is the distance table showing the approximate distance (in kilometers) between each pair of locations:\n"
    # G = generate_complete_graph(n+1)
    adj = get_adjacency_matrix(G)
    # print(adj)
    for i in range(n+1):
        if i == 0:
            temp = "Distances from Warehouse to each delivery point: "
        else:
            temp = "Distances from Delivery {name} to each delivery point: ".format(name = items[index_list[i-1]][0])
        for j in range(i,n+1):
            if i != j:
                if i == 0:
                    if j != n:
                        distance = "Warehouse to " + items[index_list[j-1]][0] + " is " + str(adj[i][j]) + " km, " 
                    else:
                        distance = "Warehouse to " + items[index_list[j-1]][0] + " is " + str(adj[i][j]) + " km.\n"
                    temp = temp + distance
                else:
                    if j != n:
                        distance = items[index_list[i-1]][0] + " to " + items[index_list[j-1]][0] + " is " + str(adj[i][j]) + " km, " 
                    else:
                        distance = items[index_list[i-1]][0] + " to " + items[index_list[j-1]][0] + " is " + str(adj[i][j]) + " km.\n"
                    temp = temp + distance
        if i != n:
            q = q + temp
    end = "Based on this distance table, we need to determine the optimal delivery route that allows the driver to start from the warehouse, visit each delivery point exactly once, and return to warehouse with the shortest possible total distance."
    q = q + "\n" + end
    return q, name_list

def get_text_ans(ans_list,name_list):
    final_list = [name_list[i] for i in ans_list]
    return final_list

def get_final_result(algo):
    if algo == "brute_force":
        ans_path, ans_distance, cost_time = tsp_brute_force(adj, start_node)
    elif algo == "tsp_brute_force_parallel":
        ans_path, ans_distance, cost_time = tsp_brute_force_parallel(adj, start_node)
    elif algo == "dynamic_programming":
        ans_path, ans_distance, cost_time = tsp_dynamic_programming(adj, start_node)
    elif algo == "tsp_dynamic_programming_numba_parallel":
        ans_path, ans_distance, cost_time = tsp_dynamic_programming_parallel(adj,start_node)
    elif algo == "branch_and_bound":
        ans_path, ans_distance, cost_time = tsp_branch_and_bound(adj, start_node)
    elif algo == "christofides_algorithm":
        ans_path, ans_distance, cost_time = christofides_algorithm(adj, start_node)
    elif algo == "greedy_algorithm":
        ans_path, ans_distance, cost_time = greedy_algorithm(adj, start_node)
    elif algo == "nearest_neighbor_algorithm":
        ans_path, ans_distance, cost_time = nearest_neighbor_algorithm(adj, start_node)
    elif algo == "simulated_annealing":
        ans_path, ans_distance, cost_time = simulated_annealing(adj, start_node)
    elif algo == "genetic_algorithm":
        ans_path, ans_distance, cost_time = genetic_algorithm(adj, start_node)
    else:
        print("未识别的算法名称！")
    ans_path_text = get_text_ans(ans_path,name_list)
    return ans_path_text,int(ans_distance),round(cost_time, 2)
if __name__ == "__main__":
    min_node_n = 3
    max_node_n = 25
    graph_n = 50
    start_node = 0
    for node in range(min_node_n,max_node_n):
        graph_save_dir = SAVE_DIR + "graph_{node}.json".format(node = node)
        graph_path = Path(graph_save_dir)
        if graph_path.exists():
            print("graph_{node}.json 已存在".format(node = node))
            continue
        data_list = []
        print("======={node} craeting=======".format(node = node))
        for graph_index in tqdm(range(graph_n), desc="Processing"):
            G = generate_complete_graph(node+1)
            adj = get_adjacency_matrix(G)
            Q , name_list = get_start_q(node,PLACE,G)
            result_list = {}
            for algo in ALGORITHM_LIST:
                ans_path_text,ans_distance,cost_time = get_final_result(algo)
                algo_dic = {
                    "ans_path_text": ans_path_text,
                    "ans_distance": ans_distance,
                    "cost_time": cost_time
                }
                result_list[str(algo)] = algo_dic
            graph_dic = {
                "graph_index": graph_index,
                "adj" : adj.tolist(),
                "name_list": name_list,
                "Q": Q,
                "result": result_list
            }
            data_list.append(graph_dic)
            path = Path(SAVE_DIR)
            if not path.exists():
                path.mkdir(parents=True)    
            with open(graph_save_dir, 'w',encoding='utf-8') as file:
                json.dump(data_list,file,ensure_ascii=False,indent=1)
    # print("---branch---")
    # print(tsp_branch_and_bound(adj,0))
