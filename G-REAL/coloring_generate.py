import networkx as nx
import random
import numpy as np
import itertools
import json
from tqdm import tqdm
import sys
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
from numba import njit, prange
SAVE_DIR = "../data/Coloring_data_50/"
ALGORITHM_LIST = [
    "backtracking",
    "greedy",
    "dsatur"
    ]
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
    q = "I am designing a public Wi-Fi network for my city, with the goal of providing free high-speed internet access across various public areas. The network will cover {n} major locations in the city: ".format(n=n)
    index_list = get_random_numbers(n)
    items = list(PLACE.items())
    name_list = [items[i][0] for i in index_list]
    all = ""
    for i in range(len(name_list)-1):
        name = name_list[i]
        if i == (len(name_list) - 2):   
            all = all + name + " and " + name_list[i+1] + ". \n"
        else:
            all = all + name + ", "
    q = q + all + "Each of these locations will have a Wi-Fi base station, but the stations are located at varying distances from one another, and some may have overlapping coverage areas. The main issue I face is how to allocate frequencies to these base stations in a way that minimizes interference. I know that if two adjacent stations use the same frequency, their signals will interfere with each other, which will affect the network’s stability and speed. \n"
    q = q + "The interference relationships between the base stations are as follows: \n"
    adj = get_adjacency_matrix(G)
    for i in range(n):
        temp = "The {name} has overlapping signal areas with ".format(name = items[index_list[i]][0])
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
    q = q + "I need to assign frequencies to the stations in such a way that no two adjacent stations use the same frequency, ensuring minimal interference. The ideal solution is to minimize the number of frequencies needed, as this would lower both the infrastructure costs and the ongoing maintenance expenses.\n"
    q = q + "Can you help me come up with a solution for frequency allocation to ensure stable and reliable network performance across all locations?\n"
    return q, name_list

def get_text_ans(ans_list,name_list):
    ans_dic = {}
    for i in range(len(name_list)):
        ans_dic[name_list[i]] = ans_list[i]
    return ans_dic

#=======回朔法=========
def graph_coloring_backtracking(graph):
    """
    使用回溯法解决图着色问题。
    :param graph: 图的邻接矩阵
    :return: 最小着色数和每个节点的颜色分配
    """
    start_time = time.time()
    def is_valid(node, c):
        """
        检查是否可以为当前节点分配颜色 c。
        :param node: 当前节点
        :param c: 当前尝试的颜色
        :return: 如果可以分配颜色 c 返回 True，否则返回 False
        """
        for neighbor in range(len(graph)):
            if graph[node][neighbor] == 1 and color[neighbor] == c:
                return False
        return True

    def backtrack(node, m):
        """
        回溯算法尝试为每个节点分配颜色。
        :param node: 当前节点
        :param m: 当前允许的最大颜色数
        :return: 如果找到合法的着色方案返回 True，否则返回 False
        """
        if node == len(graph):  # 所有节点都着色完毕
            return True

        for c in range(1, m + 1):  # 尝试每种颜色
            if is_valid(node, c):
                color[node] = c  # 分配颜色
                if backtrack(node + 1, m):  # 递归处理下一个节点
                    return True
                color[node] = 0  # 撤销分配

        return False

    n = len(graph)  # 节点数
    color = [0] * n  # 初始化每个节点的颜色

    # 从最小的颜色数开始尝试
    for m in range(1, n + 1):
        if backtrack(0, m):
            end_time = time.time()
            return m, color,end_time-start_time  # 返回最小颜色数和节点颜色分配
    end_time = time.time()
    return n, color , end_time-start_time # 最坏情况返回节点数作为颜色数
#===========贪心算法=============
def graph_coloring_greedy(graph):
    """
    使用贪心算法解决图着色问题。
    :param graph: 图的邻接矩阵
    :return: 最小着色数和每个节点的颜色分配
    """
    start_time = time.time()
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
    return min_colors, color , end_time - start_time


def graph_coloring_dsatur(graph):
    """
    使用 DSATUR 算法解决图着色问题。
    :param graph: 图的邻接矩阵
    :return: 最小着色数和每个节点的颜色分配
    """
    start_time = time.time()
    n = len(graph)  # 节点数
    colors = [-1] * n  # 初始化所有节点为未着色 (-1)
    degrees = [sum(graph[i]) for i in range(n)]  # 每个节点的度数
    saturation = [0] * n  # 每个节点的饱和度（邻居中已分配不同颜色的数量）

    def select_node():
        """
        根据饱和度和度数选择下一个节点。
        优先选择饱和度最高的节点；若饱和度相同，则选择度数最高的节点。
        """
        max_saturation = max(saturation)
        candidates = [i for i in range(n) if colors[i] == -1 and saturation[i] == max_saturation]

        if candidates:
            # 返回度数最大的节点
            return max(candidates, key=lambda x: degrees[x])
        else:
            # 如果所有节点已着色，返回 -1
            return -1

    def update_saturation(node):
        """
        更新邻居节点的饱和度。
        """
        for neighbor in range(n):
            if graph[node][neighbor] == 1 and colors[neighbor] == -1:  # 邻居未着色
                # 获取邻居的所有已着色颜色
                neighbor_colors = {colors[neighbor] for neighbor in range(n) if graph[neighbor][neighbor] == 1 and colors[neighbor] != -1}
                saturation[neighbor] = len(neighbor_colors)

    # 开始着色过程
    for _ in range(n):
        node = select_node()
        if node == -1:  # 如果所有节点都已着色，结束
            break

        # 找到可以分配的最小颜色
        neighbor_colors = {colors[neighbor] for neighbor in range(n) if graph[node][neighbor] == 1 and colors[neighbor] != -1}
        for c in range(1, n + 1):  # 从颜色1开始尝试
            if c not in neighbor_colors:
                colors[node] = c
                break

        # 更新邻居节点的饱和度
        update_saturation(node)

    # 最小着色数为分配的颜色的最大值
    min_colors = max(colors)
    end_time = time.time()
    return min_colors, colors ,end_time - start_time



def get_final_result(algo,name_list):
    if algo == "backtracking":
        min_colors, color_assignment, cost_time = graph_coloring_backtracking(adj)
    elif algo == "greedy":
        min_colors, color_assignment, cost_time = graph_coloring_greedy(adj)
    elif algo == "dsatur":
        min_colors, color_assignment, cost_time = graph_coloring_dsatur(adj)
    else:
        print("未识别的算法名称！")
    color_assignment_text = get_text_ans(color_assignment,name_list)
    return color_assignment_text,int(min_colors),round(cost_time, 2)
if __name__ == "__main__":
    min_node_n = 4
    max_node_n = 26
    graph_n = 50
    start_node = 0
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
            print(nx.is_connected(G))
            adj = get_adjacency_matrix(G)
            Q , name_list = get_start_q(node,PLACE,G)
            result_list = {}
            for algo in ALGORITHM_LIST:
                color_assignment_text,min_colors,cost_time = get_final_result(algo,name_list)
                algo_dic = {
                    "color_assignment_text": color_assignment_text,
                    "min_colors": min_colors,
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

