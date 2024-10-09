import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import pickle

# 加载并处理文本
def load_and_process_txt(file_path):
    loader = TextLoader(file_path, encoding='utf8')
    documents = loader.load()
    
    # 分割文本，保持上下文一致性
    # 一个txt看成一个文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs

# bm25 进行分词处理
def save_tokenized_docs(tokenized_docs, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(tokenized_docs, f)
def load_tokenized_docs(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def process_for_bm25(docs):
    # # 对每个文档进行分词，得到一个二维数组
    # tokenized_docs = [jieba.lcut(doc.page_content) for doc in docs]
    # # 保存
    # save_tokenized_docs(tokenized_docs, 'tokenized_docs.pkl')

    # 读取
    loaded_docs = load_tokenized_docs('tokenized_docs.pkl')

    return loaded_docs

# FAISS 向量召回
def create_faiss_index(docs):
    embeddings_model = HuggingFaceEmbeddings(model_name='/home/extra1T/model_embeding/ai-modelscope/gte-large-zh')
    vector_store = FAISS.from_documents(docs, embeddings_model)
    return vector_store

if __name__== '__main__':

    # 文档路径
    file_path = '/home/extra1T/kangh/app/kh/corpus.txt'

    # 加载并分割文本
    docs = load_and_process_txt(file_path)
    # print("dsad:",len(docs))
    # print("dsads:",docs[0].page_content)
    # 对文档进行分词，准备给 BM25 使用
    tokenized_docs = process_for_bm25(docs)

    # print("dsad:",tokenized_docs[0])

    # 初始化 BM25 模型
    bm25 = BM25Okapi(tokenized_docs)

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