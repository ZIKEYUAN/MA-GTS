import networkx as nx
import random
import numpy as np
import itertools
import json
from tqdm import tqdm
import sys
from itertools import combinations
import pulp

import time   
import matplotlib.pyplot as plt
import heapq
from queue import Queue
from pathlib import Path
import math
import multiprocessing as mp
from functools import partial
import numpy as np
SAVE_DIR = "../data/vertexcover_data_50/"
ALGORITHM_LIST = [
    "brute_force",
    "greedy",
    "2_approximation"
    ]
ALG_DIC = {
    "brute_force"
} 
COMPUTER_NAMES = {
    "Server Ravenstone": 1, "Server Windmill": 2, "Server Bluebird": 3, "Server Thunderbolt": 4, "Server Lighthouse": 5, 
    "Server Sunflower": 6, "Server Twilight": 7, "Server Falconeye": 8, "Server Redwood": 9, "Server Sandstorm": 10, 
    "Server Moonlight": 11, "Server Glacier": 12, "Server Firefly": 13, "Server Tempest": 14, "Server Bluewave": 15, 
    "Server Ironclad": 16, "Server Mirage": 17, "Server Frostbite": 18, "Server Vortex": 19, "Server Skyhawk": 20, 
    "Server Silverstone": 21, "Server Frostmoon": 22, "Server Nightshade": 23, "Server Stormbreaker": 24, "Server Firestorm": 25, 
    "Server Willowbrook": 26, "Server Thunderstrike": 27, "Server Eaglecrest": 28, "Server Oceanview": 29, "Server Blackwood": 30, 
    "Server Brightstar": 31, "Server Crimsoncloud": 32, "Server Goldleaf": 33, "Server Stealthwind": 34, "Server Darkhorse": 35, 
    "Server Emberfall": 36, "Server Silverstream": 37, "Server Windswept": 38, "Server Emberlight": 39, "Server Glacierpeak": 40, 
    "Server Shadowbrook": 41, "Server Solarflare": 42, "Server Amberwave": 43, "Server Nightfall": 44, "Server Seabreeze": 45, 
    "Server Stormcloud": 46, "Server Starfire": 47, "Server Ironbark": 48, "Server Ghostwind": 49, "Server Silverhawk": 50
}
#===============精确算法=====================
def vertex_cover_brute_force(graph):
    """
    使用暴力枚举法找到最小顶点覆盖。
    :param graph: 邻接矩阵 (二维列表)
    :return: 最小顶点覆盖 (列表)
    """
    start_time = time.time()
    n = len(graph)

    # 内部函数：检查是否是顶点覆盖
    def is_vertex_cover(vertex_set):
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i][j] == 1:  # 如果有边 (i, j)
                    if i not in vertex_set and j not in vertex_set:  # 检查是否覆盖
                        return False
        return True

    # 从小到大枚举顶点集合的大小
    for k in range(1, n + 1):
        # 枚举所有大小为 k 的顶点集合
        for vertex_set in combinations(range(n), k):
            if is_vertex_cover(set(vertex_set)):  # 将组合转换为集合传递
                end_time = time.time()
                cost_time = end_time - start_time
                return list(vertex_set),cost_time


def vertex_cover_greedy(graph):
    """
    使用贪心算法解决顶点覆盖问题。
    :param graph: 邻接矩阵 (二维列表)
    :return: 顶点覆盖集合 (列表)
    """
    start_time = time.time()
    n = len(graph)
    edges = set()  # 用于存储所有边
    vertex_cover = set()  # 顶点覆盖集合

    # 将邻接矩阵转换为边的集合
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] == 1:
                edges.add((i, j))

    # 贪心地选择顶点覆盖边
    while edges:
        # 统计每个顶点覆盖的边数
        edge_count = [0] * n
        for u, v in edges:
            edge_count[u] += 1
            edge_count[v] += 1

        # 选择覆盖最多边的顶点
        max_vertex = edge_count.index(max(edge_count))
        vertex_cover.add(max_vertex)

        # 移除与选定顶点相关的所有边
        edges = {e for e in edges if max_vertex not in e}
    end_time = time.time()
    return list(vertex_cover),end_time - start_time

def vertex_cover_2_approximation(graph):
    """
    使用 2-近似算法找到顶点覆盖。
    :param graph: 邻接矩阵 (二维列表)
    :return: 顶点覆盖的集合 (列表)
    """
    start_time = time.time()
    n = len(graph)
    visited_edges = set()  # 记录已访问的边
    vertex_cover = set()   # 保存顶点覆盖的集合

    # 遍历图的邻接矩阵，选择匹配边
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] == 1 and (i, j) not in visited_edges:
                # 如果边 (i, j) 未被访问，选择这条边的两个端点加入顶点覆盖
                vertex_cover.add(i)
                vertex_cover.add(j)
                # 标记该边为已访问
                visited_edges.add((i, j))
                visited_edges.add((j, i))
                break  # 选择一个匹配后立即跳出，避免重复选择边
    end_time = time.time()
    return list(vertex_cover),end_time - start_time

def generate_random_graph(num_nodes, r = 0.2):
    # 计算边的最小数量
    min_edges = int(num_nodes ** 2 * r)

    # 创建一个空的无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(range(num_nodes))

    # 首先，确保图是连通的
    # 使用生成树来确保连通性，这里我们可以用 NetworkX 提供的最小生成树方法
    tree = nx.minimum_spanning_tree(nx.gnm_random_graph(num_nodes, num_nodes-1))
    G.add_edges_from(tree.edges())

    # 添加更多的边，直到边的数量满足要求
    while len(G.edges()) < min_edges:
        # 随机选择两个不同的节点添加一条边
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v)

    return G

def plot_and_save_graph(G, filename="random_graph.png"):
    # 使用 NetworkX 提供的绘图功能画出图
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # 布局方法，spring_layout 更适合大多数图
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Random Graph Visualization")
    
    # 保存图像到指定文件
    plt.savefig(filename, format="PNG")
    print(f"图像已保存为 {filename}")
    plt.close()

def print_graph(G):
    for (u, v, wt) in G.edges(data='weight'):
        print(f"从节点 {u} 到节点 {v} 的边，权重为 {wt:.2f}")

def get_adjacency_matrix(G):
    n = G.number_of_nodes()
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for (u, v) in G.edges():
        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1  # 因为是无向图，矩阵是对称的
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
    q = "Our company has {n} computers connected by several communication links. These computers are named: ".format(n=n)
    index_list = get_random_numbers(n)
    items = list(COMPUTER_NAMES.items())
    name_list = [items[i][0] for i in index_list]
    all = ""
    for i in range(len(name_list)-1):
        name = name_list[i]
        if i == (len(name_list) - 2):   
            all = all + name + " and " + name_list[i+1] + ". \n"
        else:
            all = all + name + ", "
    q = q + all + "To ensure network security, we need to install monitoring devices (such as firewalls or intrusion detection systems) on some of these computers so that all communication links are monitored. \n"
    q = q + "Assume that the connections between the computers (i.e., the communication links) are bidirectional. This means that information can flow in both directions across any link. Our goal is to deploy monitoring devices in a way that ensures all communication links are covered by at least one monitoring device. \n"
    q = q + "Problem: How can we select the minimum number of computers to deploy monitoring devices, such that every communication link is monitored by at least one device? \n"
    q = q + "Communication links as follows: : \n"
    adj = get_adjacency_matrix(G)
    for i in range(n):
        temp = "{name} is connected with ".format(name = items[index_list[i]][0])
        title = temp
        for j in range(i,n):
            if adj[i][j] == 1:
                temp = temp + items[index_list[j]][0] + ", "
        # words = temp.split() 
        # last_word = words[-1]
        # if last_word != "with":
        temp = temp[:-2] + ". \n"
        words = temp.split() 
        last_word = words[-1]
        if last_word != "wit.":
            q = q + temp
    return q, name_list

def get_text_ans(ans_list,name_list):
    final_list = [name_list[i] for i in ans_list]
    return final_list





def get_final_result(algo,name_list):
    if algo == "brute_force":
        vertex_cover, cost_time = vertex_cover_brute_force(adj)
    elif algo == "greedy":
        vertex_cover, cost_time = vertex_cover_greedy(adj)
    elif algo == "2_approximation":
        vertex_cover, cost_time = vertex_cover_2_approximation(adj)
    else:
        print("未识别的算法名称！")
    vertex_cover_text = get_text_ans(vertex_cover,name_list)
    return vertex_cover_text,round(cost_time, 2)
if __name__ == "__main__":
    min_node_n = 4
    max_node_n = 26
    graph_n = 50

    for node in range(min_node_n,max_node_n):
        graph_save_dir = SAVE_DIR + "graph_{node}.json".format(node = node)
        graph_path = Path(graph_save_dir)
        # if graph_path.exists():
        #     print("graph_{node}.json 已存在".format(node = node))
        #     continue
        data_list = []
        print("======={node} craeting=======".format(node = node))
        for graph_index in tqdm(range(graph_n), desc="Processing"):
            G = generate_random_graph(node)
            # print(nx.is_connected(G))
            adj = get_adjacency_matrix(G)
            Q , name_list = get_start_q(node,COMPUTER_NAMES,G)
            # print(adj)
            # print(name_list)
            # print(Q)
            result_list = {}
            for algo in ALGORITHM_LIST:
                vertex_cover,cost_time = get_final_result(algo,name_list)
                # print(vertex_cover)
                algo_dic = {
                    "vertex_cover_text": vertex_cover,
                    "min_vertex": len(vertex_cover),
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

