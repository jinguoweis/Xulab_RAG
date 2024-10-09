# 加载文档和 BM25/FAISS 初始化
from tqdm import tqdm
import json
import pickle
import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import csv
from langchain.docstore.document import Document
from rrf_fusion_phj import rrf_fusion
from bge_reanker import bge_rerank
from utils_process import load_and_process_txt
from utils_process import *
# with open('/home/extra1T/jin/app/docs.pkl', 'rb') as f:
#     docs = pickle.load(f)
#加载保存好的 bm25

embeddings_model = HuggingFaceEmbeddings(model_name='/home/extra1T/model_embeding/ai-modelscope/gte-large-zh')
bm25_model_file = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/model.pkl'
# 文档路径
file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt'
# 加载并分割文本
docs = load_and_process_txt(file_path)
bm25 = load_bm25_model(bm25_model_file)
save_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_faiss'
vector_store = load_faiss_index(save_path, embeddings_model)

# 假设 questions.csv 文件路径为 '/path/to/questions.csv'
questions_file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/test_question.csv'
questions = load_questions(questions_file_path)


final_contexts = process_questions_with_progress(questions, docs, bm25, vector_store)
# 输出 final_contexts，显示多少问题得到了上下文信息
print(f"共处理了 {len(final_contexts)} 个问题，所有问题的上下文已完成召回与重排序。")
file_path1 = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/final_contexts_question.json'

# 将 final_contexts 写入 JSON 文件
with open(file_path1, 'w', encoding='utf-8') as f:
    json.dump(final_contexts, f, ensure_ascii=False, indent=4)

print(f"数据已保存至 {file_path1}")