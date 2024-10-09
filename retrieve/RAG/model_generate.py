import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
from utils_process import *
import csv
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import PromptTemplate
import json

# 定义文件路径
json_file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/final_contexts_question.json'

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    final_contexts = json.load(file)
    # final_contexts = final_contexts[0:100]
# 假设你的 LLM 已经初始化好了
# llm = LLM()  # 这里的 LLM 实际上是你的语言模型实例

# prompt_template = """你是一个用于问答任务的助手。
# 使用下面检索到的上下文片段来完整的回答用户的问题。
# 如果你不知道答案，只需说你不知道。
# {context}

# 问题: {question}
# 回答:"""
prompt_template = """你是一个专门负责问答任务的智能助手。

你需要两个步骤来执行这个任务：
1. 判断检索到的内容中是否包含问题的答案，不包含则只能输出：“结合给定的资料无法回答这个问题。”请参考我给出的示例：
<example>

用户问题:
为什么深圳的高分考生更青睐北大和清华？
检索到的上下文:
清华大学成立于1911年,最初为留美预备学校,后来发展为综合性大学。北京大学成立于1898年前身为京师大学堂,是中国现代高等教育的发端。
回答:
结合给定的资料无法回答这个问题。

用户问题:
小明今天晚上的晚餐是什么？
检索到的上下文:
小明喜欢吃各种美食，尤其是川菜和粤菜。他最爱吃麻辣鲜香的火锅，总是点满满一桌的毛肚、黄喉、牛肉片和鸭血。他也喜欢红烧肉，觉得那肥瘦相间的肉块入口即化，配上白米饭再完美不过。
回答:
结合给定的资料无法回答这个问题。

</example>

2. 如果检索到的内容包含问题的答案，请根据以下检索到的上下文片段回答用户的问题。要求答案简洁、直接，避免冗长或不必要的信息。
回答时，请始终包括问题中的主语，并确保前后逻辑连贯。
请参考我给出的示例：
<example>

用户问题:
东京街头最抢手的时尚单品是什么？
检索到的上下文:
热荐:6月东京街头角斗士鞋(组图)导读：在东京，最抢手的非角斗士鞋莫属了。
回答：
东京街头最抢手的时尚单品是角斗士鞋。

用户问题:
德国科隆最著名的标志是什么？
检索到的上下文:
全球14座名城夜景之德国科隆德国·科隆科隆大教堂和它的157米的两个尖顶已经成为科隆这个城市最著名的标志。
回答：
德国科隆最著名的标志是科隆大教堂及其157米高的双尖顶。

</example>

请严格按照给定的示例逻辑结构进行回答，确保问题的主语清晰出现。
必须根据检索到的内容进行回答，不要根据你自身的知识储备进行回答，如果检索到的内容中不包含问题对应的知识点，请务必直接回答“结合给定的资料无法回答这个问题。”这句话。

检索到的上下文：{context}

用户问题：{question}
回答："""


path = '/home/extra1T/model_embeding/qwen/Qwen2-7B'
llm = load_model(path)
# 定义 Prompt
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# 构建链条
rag_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# 准备输出文件
output_file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/answers.csv'
header = ['Question', 'Answer']

# 存储问题和答案到 CSV 文件
with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # 写入表头

    # tqdm 用于显示进度条
    for entry in tqdm(final_contexts, desc="处理问题", unit="个"):
        query = entry['query']
        context = entry['context']
        
        # 将问题和上下文传递给大模型
        raw_answer = rag_chain.run({"context": context, "question": query})

        # 将问题和答案写入到 CSV 文件
        writer.writerow([query, raw_answer])

print(f"所有问题和答案已经写入到 {output_file_path}")

drop(output_file_path)