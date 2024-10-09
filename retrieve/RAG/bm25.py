import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
# 加载并处理文本
import pickle
# import sys
# sys.path.append('/home/extra1T/jin/GoMate')
from utils_process import *

# 文档路径
file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus.txt'

# 加载并分割文本
docs = load_and_process_txt1(file_path)
# print("dsad:",len(docs))
# print("dsads:",docs[0].page_content)
# 对文档进行分词，准备给 BM25 使用
tokenized_docs = process_for_bm25(docs)
# 初始化 BM25 模型
bm25 = BM25Okapi(tokenized_docs)
# 保存 BM25 模型
bm25_model_file = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/bm25_model.pkl'
save_bm25_model(bm25, bm25_model_file)
# 输入查询，查询的内容需要分词
query = "第8届南方新丝路模特大赛的招募活动是在哪里举行的？"
query_tokens = jieba.lcut(query)

# 使用 BM25 进行检索，得到每个文档的相关性分数
doc_scores = bm25.get_scores(query_tokens)
print(len(doc_scores))

# 设置要返回的 topk 个文档
topk = 3  # 例如返回前3个相似答案

# 获取分数最高的 topk 个文档索引
topk_indices = doc_scores.argsort()[-topk:][::-1]

# 获取分数最高的 topk 个文档内容
topk_matches = [docs[idx].page_content for idx in topk_indices]
# 输出 topk 个最相关的文档内容
print("最相关的文档内容：")
for i, match in enumerate(topk_matches, 1):
    print(f"Top {i}:\n{match}\n")


################测试#########################
# bm25_model_file = '/home/extra1T/jin/app/bm25_model.pkl'
# bm25 = load_bm25_model(bm25_model_file)
# query = "四川金顶为什么被证监会成都稽查局立案调查？"
# query_tokens = jieba.lcut(query)

# # 使用 BM25 进行检索，得到每个文档的相关性分数
# doc_scores = bm25.get_scores(query_tokens)

# # 设置要返回的 topk 个文档
# topk = 3  # 例如返回前3个相似答案

# # 获取分数最高的 topk 个文档索引
# topk_indices = doc_scores.argsort()[-topk:][::-1]

# # 获取分数最高的 topk 个文档内容
# topk_matches = [docs[idx].page_content for idx in topk_indices]

# # 输出 topk 个最相关的文档内容
# print("最相关的文档内容：")
# for i, match in enumerate(topk_matches, 1):
#     print(f"Top {i}:\n{match}\n")