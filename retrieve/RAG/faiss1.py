import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
from utils_process import *
topk =5 
# 载入并处理文本

# 文档路径
file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt'

# 加载并分割文本
docs = load_and_process_txt1(file_path)

# 初始化 FAISS 向量检索
vector_store = create_faiss_index(docs)

query = "第8届南方新丝路模特大赛的招募活动是在哪里举行的？"

# 使用 FAISS 进行检索，并返回文档和对应的索引编号
result_with_score = vector_store.similarity_search_with_score(query, k=topk)
faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]  # 直接访问 page_content

print("FAISS 检索结果：")
for i, (content, idx) in enumerate(faiss_topk_matches, 1):
    print(f"Top {i}:\nContent: {content}\nIndex: {idx}\n")

######################################将新闻按照每行来单独切分############################################### 
# # 文档路径
# file_path = '/home/extra1T/jin/app/corpus.txt'

# # 加载并分割文本
# docs = load_and_process_txt(file_path)

# # 初始化 FAISS 向量检索
# vector_store = create_faiss_index(docs)
# save_faiss_index(vector_store,'/home/extra1T/jin/GoMate')
# query = "第8届南方新丝路模特大赛的招募活动是在哪里举行的？"

# # 使用 FAISS 进行检索，并返回文档和对应的索引编号
# topk = 5  # 定义要检索的前K个结果
# result_with_score = vector_store.similarity_search_with_score(query, k=topk)
# faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]  # 直接访问 page_content

# print("FAISS 检索结果：")
# for i, (content, idx) in enumerate(faiss_topk_matches, 1):
#     print(f"Top {i}:\nContent: {content}\nIndex: {idx}\n")
    