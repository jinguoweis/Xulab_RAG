import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
from utils_process import *

# 文档路径
file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt'

# 加载并分割文本
docs = load_and_process_txt1(file_path)

# 处理文档并进行 BM25 分词
tokenized_docs = process_for_bm25(docs)

# 初始化 BM25 模型
bm25 = BM25Okapi(tokenized_docs)

# 初始化 FAISS 向量检索
vector_store = create_faiss_index(docs)

# 输入查询，查询内容需要分词
query = "第8届南方新丝路模特大赛的招募活动是在哪里举行的？"
query_tokens = jieba.lcut(query)

# 使用 BM25 进行检索
bm25_doc_scores = bm25.get_scores(query_tokens)

# 设置返回的 topk 个文档
topk = 3

# 获取 BM25 topk 文档和其索引编号
bm25_topk_indices = bm25_doc_scores.argsort()[-topk:][::-1]
bm25_topk_matches = [(docs[idx].page_content, idx) for idx in bm25_topk_indices]  # 返回内容和索引

# 使用 FAISS 进行检索，并返回文档和对应的索引编号
result_with_score = vector_store.similarity_search_with_score(query, k=topk)
faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]  # 直接访问 page_content

# 打印结果
print("BM25 检索结果：")
for i, (content, idx) in enumerate(bm25_topk_matches, 1):
    print(f"Top {i}:\nContent: {content}\nIndex: {idx}\n")

print("FAISS 检索结果：")
for i, (content, idx) in enumerate(faiss_topk_matches, 1):
    print(f"Top {i}:\nContent: {content}\nIndex: {idx}\n")

# 返回结果
bm25_topk_matches, faiss_topk_matches