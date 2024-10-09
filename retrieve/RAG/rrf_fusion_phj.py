import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import pickle
from langchain.docstore.document import Document
import sys
# sys.path.append('/home/extra1T/jin/GoMate/RAG')
from utils_process import *
# from utils_process import load_and_process_txt

# 载入并处理文本

file_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt'
docs = load_and_process_txt(file_path)

# # 使用langchain框架进行加载并分割文本  此种加载方式 会和其他新闻有重叠 造成一部分的重叠 
# file_path = '/home/extra1T/jin/app/corpus.txt'
# docs = load_and_process_txt1(file_path)

bm25_model_file = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/model.pkl'
bm25 = load_bm25_model(bm25_model_file)
# with open('/home/extra1T/jin/app/docs.pkl', 'rb') as f:
#     docs1 = pickle.load(f)
embeddings_model = HuggingFaceEmbeddings(model_name='/home/extra1T/model_embeding/ai-modelscope/gte-large-zh')
save_path = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_faiss'
vector_store = load_faiss_index(save_path, embeddings_model)

# 输入查询，查询内容需要分词
query = "四川金顶为什么被证监会成都稽查局立案调查？"
query_tokens = jieba.lcut(query)

# 使用 BM25 进行检索
bm25_doc_scores = bm25.get_scores(query_tokens)

# 设置返回的 top K 个文档
# topk = 60
topk = 5 
# 获取 BM25 top K 文档和其索引编号
bm25_topk_indices = bm25_doc_scores.argsort()[-topk:][::-1]
bm25_topk_matches = [(docs[idx].page_content, idx) for idx in bm25_topk_indices]

# 使用 FAISS 进行检索，并返回文档和对应的索引编号
result_with_score = vector_store.similarity_search_with_score(query, k=topk)
faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]  # 直接访问 page_content
# 不在 BM25 检索时限制 top K，获取全部文档的评分和索引

# bm25_topk_matches
# faiss_topk_matches
# 使用 RRF 进行分数融合
rrf_results = rrf_fusion(bm25_topk_matches, faiss_topk_matches, k=60)
topk = 3
# 输出最终融合后的 top K 结果
print("RRF 融合结果：")
for idx, (doc_idx, score) in enumerate(rrf_results[:topk], 1):
    print(f"Top {idx}:\nContent: {docs[doc_idx].page_content}\nRRF Score: {score}\n")
