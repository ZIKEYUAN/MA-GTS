from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.prompts import PromptTemplateGenerator
from camel.types import ModelType, OpenAIBackendRole
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, TaskType
from camel.utils import OpenAITokenCounter
from camel.embeddings import OpenAIEmbedding
from camel.storages import QdrantStorage
from camel.retrievers import AutoRetriever,VectorRetriever
from camel.types import EmbeddingModelType
from camel.memories import (
    ChatHistoryBlock,
    LongtermAgentMemory,
    MemoryRecord,
    ScoreBasedContextCreator,
    VectorDBBlock,
)
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.toolkits import (
    SearchToolkit,
    MathToolkit,
)
import time
import argparse
from pathlib import Path
import json
from camel.toolkits import FunctionTool
from prompt.algo_code import tsp_dynamic_programming,tsp_greedy_nearest_neighbor,graph_coloring_backtracking,vertex_cover_brute_force
from camel.loaders import create_file_from_raw_bytes
import prompt.sys_prompt.sys_prompt as pt
import os
import prompt.algo_base_easy as alb
import json
import io
OPENAI_API_KEY = ""
DEEPSEEK_KEY = ""
CLOSEAI_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
DATA_DIR = ""
SAVE_RESULT_DIR = ""
PROBLEM_LIST = ["TSP_Problem","Coloring_Problem"]
DATA_LIST = [
    "TSP Graph"
    ]

def generate_adjacency_list(nodes,edges):
    r"""Generates an adjacency list and node mapping from the given graph input.

    Args:
        nodes (list): A list of node names (strings).
        edges (list of tuples): A list of edges, where each edge is a tuple
            (source_node, target_node, attributes). Attributes is a dictionary containing
            additional information about the edge (e.g., weight, length).

    Returns:
        dict: A dictionary containing:
            - "adjacency_list" (dict): The adjacency list where the keys are node indices
              and the values are lists of tuples representing the neighbors and edge weights.
            - "node_mapping" (dict): A dictionary mapping node names to numeric indices.
    """
    # nodes = graph_input["nodes"]
    # edges = graph_input["edges"]
    # graph_type = graph_input["graph_type"]

    # 创建节点映射，将节点名称映射为数字索引
    node_mapping = {node: index for index, node in enumerate(nodes)}

    # 初始化邻接表
    adjacency_list = {index: [] for index in range(len(nodes))}

    # 填充邻接表
    for edge in edges:
        if len(edge) == 3:
            source, target, attributes = edge
        else:
            source, target = edge
            attributes = {"weight":1}
        try:
            source_index = node_mapping[source]
            target_index = node_mapping[target]
        except Exception as e:
            source = "Node " + str(source)
            target = "Node " + str(target)
            source_index = node_mapping[source]
            target_index = node_mapping[target]
        # 获取权重值，默认为 1，尝试从 attributes 中找到第一个数值属性
        weight = next((value for key, value in attributes.items() if isinstance(value, (int, float))), 1)
        
        # 添加边到邻接表
        adjacency_list[source_index].append((target_index, weight))

        # 如果是无向图，还需要添加反向边
        # if "un" in graph_type:
        #     adjacency_list[target_index].append((source_index, weight))

    # 生成输出结果
    output = {
        "adjacency_list": adjacency_list,
        "node_mapping": node_mapping,
    }

    return output

def read_json_files_in_directory(directory_path):
    # 获取目录下所有文件的文件名
    try:
        file_names = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"目录 {directory_path} 未找到！")
        return
    
    json_files = [file for file in file_names if file.endswith('.json')]
    
    if not json_files:
        print("目录中没有找到任何 JSON 文件！")
        return
    
    # 逐个读取每个 JSON 文件的内容
    json_data_list = []
    for json_file in json_files:
        json_file_path = os.path.join(directory_path, json_file)
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data_list.append({
                    "file_name": json_file,
                    "content": data
                })
                print(f"成功读取文件: {json_file}")
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")
    
    return json_data_list

def save_result(problem_type,save_dir,result_list,file_name,model_name):
    save_dir_final = SAVE_RESULT_DIR + model_name + "/" + problem_type + "/" + save_dir + "/" + file_name
    path = Path(SAVE_RESULT_DIR + model_name + "/" + problem_type + "/" + save_dir + "/")
    if not path.exists():
        path.mkdir(parents=True)  
    with open(save_dir_final, 'w',encoding='utf-8') as file:
        json.dump(result_list,file,ensure_ascii=False,indent=1)

def get_pdf_bytesio(file_path: str) -> io.BytesIO:
    with open(file_path, "rb") as f:
        # Read the content of the file into a bytes object
        file_bytes = f.read()
    
    # Convert the bytes object into a BytesIO stream
    return io.BytesIO(file_bytes)

def data_parse(file_stream):
    from camel.loaders.unstructured_io import UnstructuredIO

    file_bytes = file_stream.getvalue()
    file_obj = create_file_from_raw_bytes(file_bytes, filename="filename.pdf")
    file_elements = []
    page_content = ""
    for page in file_obj.docs:
        page_content = page_content + page['page_content']
    element = UnstructuredIO().create_element_from_text(
        text=page_content,
        # file_directory="filepath",
        # filename="filename",
        # filetype="filetype",
    )
        # file_elements.append(element)
    return element

def generate_text_question(text_information,problem_information):
    user_q = """
    **Text Information Dictionary**
    {text_information}
    
    **Problem Information Dictionary**
    {problem_information}
    """.format(text_information = text_information, problem_information = problem_information )
    return user_q

def generate_tool_use_text(algorithm_information,graph_information):
    user_q = """
    **Algorithm Information Dictionary**
    {algorithm_information}
    
    **Graph Information Dictionary**
    {graph_information}
    """.format(algorithm_information = algorithm_information, graph_information = graph_information )
    return user_q

def reasoning(problem_type,Q,agent_list,graph_index,file_name,adj,name_list,result_dic,model_name,real_result):
    start_time = time.time()
    # Q = "This is a undirected graph with the following edges:\nFrom node 0 to node 1, distance is 5\nFrom node 0 to node 2, distance is 2\nFrom node 0 to node 3, distance is 3\nFrom node 0 to node 4, distance is 4\nFrom node 0 to node 5, distance is 1\nFrom node 0 to node 6, distance is 4\nFrom node 0 to node 7, distance is 3\nFrom node 0 to node 8, distance is 1\nFrom node 0 to node 9, distance is 3\nFrom node 0 to node 10, distance is 3\nFrom node 0 to node 11, distance is 2\nFrom node 0 to node 12, distance is 5\nFrom node 0 to node 13, distance is 3\nFrom node 0 to node 14, distance is 2\nFrom node 1 to node 2, distance is 5\nFrom node 1 to node 3, distance is 2\nFrom node 1 to node 4, distance is 3\nFrom node 1 to node 5, distance is 4\nFrom node 1 to node 6, distance is 1\nFrom node 1 to node 7, distance is 5\nFrom node 1 to node 8, distance is 3\nFrom node 1 to node 9, distance is 4\nFrom node 1 to node 10, distance is 1\nFrom node 1 to node 11, distance is 1\nFrom node 1 to node 12, distance is 5\nFrom node 1 to node 13, distance is 2\nFrom node 1 to node 14, distance is 4\nFrom node 2 to node 3, distance is 4\nFrom node 2 to node 4, distance is 1\nFrom node 2 to node 5, distance is 4\nFrom node 2 to node 6, distance is 4\nFrom node 2 to node 7, distance is 3\nFrom node 2 to node 8, distance is 5\nFrom node 2 to node 9, distance is 3\nFrom node 2 to node 10, distance is 5\nFrom node 2 to node 11, distance is 1\nFrom node 2 to node 12, distance is 4\nFrom node 2 to node 13, distance is 3\nFrom node 2 to node 14, distance is 4\nFrom node 3 to node 4, distance is 1\nFrom node 3 to node 5, distance is 2\nFrom node 3 to node 6, distance is 1\nFrom node 3 to node 7, distance is 2\nFrom node 3 to node 8, distance is 3\nFrom node 3 to node 9, distance is 2\nFrom node 3 to node 10, distance is 3\nFrom node 3 to node 11, distance is 3\nFrom node 3 to node 12, distance is 5\nFrom node 3 to node 13, distance is 2\nFrom node 3 to node 14, distance is 4\nFrom node 4 to node 5, distance is 4\nFrom node 4 to node 6, distance is 4\nFrom node 4 to node 7, distance is 1\nFrom node 4 to node 8, distance is 4\nFrom node 4 to node 9, distance is 2\nFrom node 4 to node 10, distance is 3\nFrom node 4 to node 11, distance is 3\nFrom node 4 to node 12, distance is 1\nFrom node 4 to node 13, distance is 1\nFrom node 4 to node 14, distance is 2\nFrom node 5 to node 6, distance is 1\nFrom node 5 to node 7, distance is 3\nFrom node 5 to node 8, distance is 1\nFrom node 5 to node 9, distance is 1\nFrom node 5 to node 10, distance is 3\nFrom node 5 to node 11, distance is 2\nFrom node 5 to node 12, distance is 2\nFrom node 5 to node 13, distance is 2\nFrom node 5 to node 14, distance is 4\nFrom node 6 to node 7, distance is 3\nFrom node 6 to node 8, distance is 3\nFrom node 6 to node 9, distance is 1\nFrom node 6 to node 10, distance is 5\nFrom node 6 to node 11, distance is 1\nFrom node 6 to node 12, distance is 3\nFrom node 6 to node 13, distance is 5\nFrom node 6 to node 14, distance is 3\nFrom node 7 to node 8, distance is 3\nFrom node 7 to node 9, distance is 3\nFrom node 7 to node 10, distance is 1\nFrom node 7 to node 11, distance is 1\nFrom node 7 to node 12, distance is 1\nFrom node 7 to node 13, distance is 5\nFrom node 7 to node 14, distance is 2\nFrom node 8 to node 9, distance is 3\nFrom node 8 to node 10, distance is 3\nFrom node 8 to node 11, distance is 3\nFrom node 8 to node 12, distance is 1\nFrom node 8 to node 13, distance is 1\nFrom node 8 to node 14, distance is 2\nFrom node 9 to node 10, distance is 3\nFrom node 9 to node 11, distance is 1\nFrom node 9 to node 12, distance is 2\nFrom node 9 to node 13, distance is 2\nFrom node 9 to node 14, distance is 4\nFrom node 10 to node 11, distance is 4\nFrom node 10 to node 12, distance is 2\nFrom node 10 to node 13, distance is 5\nFrom node 10 to node 14, distance is 1\nFrom node 11 to node 12, distance is 2\nFrom node 11 to node 13, distance is 3\nFrom node 11 to node 14, distance is 2\nFrom node 12 to node 13, distance is 3\nFrom node 12 to node 14, distance is 1\nFrom node 13 to node 14, distance is 4.The goal is to find the shortest possible route that visits each node exactly once and returns to the starting node.Please determine the optimal solution for this Traveling Salesman Problem (TSP).You can use Nearest Neighbor Algorithm solve this problem. Provide the sequence of nodes that form this shortest route and the total distance of this route.Start from node 0."
    last_assistant_response = ""
    real_Q = Q
    text_information = ""
    graph_information = ""
    problem_information = ""
    temp_q = ""
    algorithm_information = ""
    graph_structured_information = ""
    try_num = 0 
    try_limited = 5   
    for singel_agent in agent_list:
        graph_result = {
            "file_name":file_name,
            "graph_index":graph_index,
            "adj":adj,
            "name_list":name_list,
            "question":real_Q.content,
            "real_result":real_result
        }
        agent_name = singel_agent.role_name
        print("=======================")
        print(agent_name + " chat_index:{try_num}".format(try_num = try_num ) + "\n正在处理中！\n")
        singel_agent.reset()          
        while try_num < try_limited:
            try:
                if agent_name == "Structured_Graph_Information_Agent":
                    assistant_response = singel_agent.step(temp_q)
                elif agent_name == "Algorithm_Solving_Agent":
                    temp_q = generate_tool_use_text(algorithm_information,graph_structured_information)
                    assistant_response = singel_agent.step(temp_q)
                else:
                    assistant_response = singel_agent.step(Q)
                break
            except Exception as e:
                print(e)
                print("reload agent!")
                print(agent_name + " chat_index:{try_num}".format(try_num = try_num ) + "\n正在处理中！\n")
                try_num = try_num + 1
                # time.sleep(10)
        if try_num == try_limited:
            graph_result["response"] = "**ERROR RESULT**"
            result_dic[agent_name].append(graph_result)
            save_result(problem_type,agent_name,result_dic[agent_name],file_name,model_name)
            continue
                
        print(agent_name + "生成的回答！\n")
        print(assistant_response.msg.content)
        end_time = time.time()
        total_time = end_time - start_time
        graph_result["response"] = assistant_response.msg.content
        graph_result["use_time"] = total_time
        result_dic[agent_name].append(graph_result)
        save_result(problem_type,agent_name,result_dic[agent_name],file_name,model_name)
        print("=======================")
        if agent_name in ["Textual_Information_Extraction_Agent",
                          "Graph_Structure_Information_Extraction_Agent",
                          "Problem_Information_Extraction_Agent"]:
            Q = real_Q
            if agent_name == "Textual_Information_Extraction_Agent":
                text_information = assistant_response.msg.content
            elif agent_name == "Graph_Structure_Information_Extraction_Agent":
                graph_information = assistant_response.msg.content
                temp_q = BaseMessage.make_user_message(role_name=agent_name, content=graph_information)
            elif agent_name == "Problem_Information_Extraction_Agent":
                problem_information = assistant_response.msg.content
                text_q = generate_text_question(text_information,problem_information)
                Q = BaseMessage.make_user_message(role_name=agent_name, content=text_q)
        elif agent_name == "Structured_Graph_Information_Agent":
            graph_structured_information = assistant_response.msg.content
        elif agent_name == "Graph_Theory_Agent":
            algorithm_information = assistant_response.msg.content
        last_assistant_response = assistant_response
        # else:
        #     Q = BaseMessage.make_user_message(role_name=agent_name, content=assistant_response.msg.content)

def get_algo_base_memory():
    memory = LongtermAgentMemory(
        context_creator=ScoreBasedContextCreator(
            token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
            token_limit=40960,
        ),
        chat_history_block=ChatHistoryBlock(),
        vector_db_block=VectorDBBlock(),
    )
    # Create and write new records
    records = [
        MemoryRecord(
            message=BaseMessage.make_assistant_message(
                role_name="Graph theory base",
                content=alb.ALGO_BASE + "/n/n" + pt.AGENT_GTA_SYS_PROMPT
            ),
            role_at_backend=OpenAIBackendRole.ASSISTANT,
        ),
        # MemoryRecord(
        #     message=BaseMessage.make_assistant_message(
        #         role_name="Agent",
        #         content=pt.AGENT_GTA_SYS_PROMPT
        #     ),
        #     role_at_backend=OpenAIBackendRole.ASSISTANT,
        # ),
    ]
    memory.write_records(records)
    return memory
def create_base_model(model_name="deepseek_v3"):
    if model_name == "qwen2.5-14b":
        base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_2_5_14B
                )
    elif model_name == "gpt4o-mini":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4O_MINI
                )   
    elif model_name == "qwen2.5-7b":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_2_5_7B
                )   
    elif model_name == "qwen2.5-72b":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_2_5_72B
                )   
    elif model_name == "qwen-turbo":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_TURBO
                ) 
    elif model_name == "qwen-qwq32b":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.QWEN,
                    model_type=ModelType.QWEN_QWQ_32B
                ) 
    elif model_name == "gpt3.5":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_3_5_TURBO
                )       
    elif model_name == "deepseek_v3":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.DEEPSEEK,
                    model_type=ModelType.DEEPSEEK_CHAT,
                    api_key =  DEEPSEEK_KEY
                )   
    elif model_name == "deepseek_v3_closeai":
         base_model = ModelFactory.create(
                    model_platform=ModelPlatformType.DEEPSEEK,
                    model_type=ModelType.DEEPSEEK_CHAT,
                    url = 'https://api.openai-proxy.org/v1',
                    api_key =  CLOSEAI_KEY)            
    return base_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model select")
    available_models = [
        "gpt3.5",
        "gpt4o-mini",
        "qwen2.5-14b",
        "qwen2.5-7b",
        "qwen2.5-72b",
        "qwen-turbo",
        "qwen-qwq32b",
        "deepseek_v3",
        "deepseek_v3_closeai"
    ]
    parser.add_argument(
        '--model_name',
        type=str,
        choices=available_models,
        default='deepseek_v3',
        help="Specify the model name. Available options: {}".format(', '.join(available_models))
    )
    args = parser.parse_args()
    model_name = args.model_name
    
    for data_name in DATA_LIST:
        print(data_name)
        directory_path = DATA_DIR + data_name
        json_data_list = read_json_files_in_directory(directory_path=directory_path)
        tool_code = [FunctionTool(func) for func in [generate_adjacency_list]]
        graph_algo = [FunctionTool(func) for func in [
                                                    #   tsp_dynamic_programming,
                                                      tsp_greedy_nearest_neighbor,
                                                      graph_coloring_backtracking,
                                                      vertex_cover_brute_force,
                                                      ]
                      ]
        for file in json_data_list:
            save_dir_final = SAVE_RESULT_DIR + "/" + model_name +"/" + data_name + "/Algorithm_Solving_Agent/" + file["file_name"]
            path = Path(save_dir_final)
            if path.exists():
                print(file["file_name"]+"存在！跳过！")
                continue
            print("File name:",file["file_name"])
            graph_data = file["content"]
            result_dic = {
                "Textual_Information_Extraction_Agent":[],
                "Graph_Structure_Information_Extraction_Agent":[],
                "Problem_Information_Extraction_Agent":[],
                "Structured_Graph_Information_Agent":[],
                "Graph_Theory_Agent":[],
                "Algorithm_Solving_Agent":[],
            }
            graph_index = 0
            for graph in graph_data:
                graph_description = graph["graph_description"]
                Q_context = graph_description + graph["QA"]["specail_qa"]["Q"]
                agent_list = []
                base_model1 = create_base_model(model_name)
                base_model2 = create_base_model(model_name)
                base_model3 = create_base_model(model_name)  
                base_model4 = create_base_model(model_name)
                base_model5 = create_base_model(model_name)
                base_model6 = create_base_model(model_name)
                question_msg = BaseMessage.make_user_message(role_name="Question", content=Q_context)

                agent_TIEA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Textual_Information_Extraction_Agent",
                        content=pt.TIEA_SYS_PROMPT,
                    ),
                    model=base_model1,
                )
                
                agent_GSIEA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Graph_Structure_Information_Extraction_Agent",
                        content=pt.GSIEA_SYS_PROMPT,
                    ),
                    model=base_model2,
                )
                
                agent_PIEA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Problem_Information_Extraction_Agent",
                        content=pt.PIEA_SYS_PROMPT,
                    ),
                    model=base_model3,
                )
                
                agent_GTA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Graph_Theory_Agent",
                        content=pt.GTA_SYS_PROMPT + "\n **Graph Theory Algorithm Library:** \n" + alb.ALGO_BASE ,
                    ),
                    model=base_model4,
                )
                
                agent_SGIA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Structured_Graph_Information_Agent",
                        content=pt.SGIA_SYS_PROMPT,
                    ),
                    model=base_model5,
                    tools=tool_code
                )
                
                agent_ASA = ChatAgent(
                    system_message=BaseMessage.make_assistant_message(
                        role_name="Algorithm_Solving_Agent",
                        content=pt.AGENT_ASA_SYS_PROMPT,
                    ),
                    model=base_model6,
                    tools=graph_algo
                )
                
                
                agent_list.append(agent_TIEA)
                agent_list.append(agent_GSIEA)
                agent_list.append(agent_PIEA)
                agent_list.append(agent_GTA)
                agent_list.append(agent_SGIA)
                agent_list.append(agent_ASA)
                reasoning(data_name,question_msg,agent_list,int(graph_index),file["file_name"],None,None,result_dic,model_name,graph["QA"]["specail_qa"]["A"])
                graph_index = graph_index + 1
        #         break
        #     break
        # break