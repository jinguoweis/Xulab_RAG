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
# from rrf_fusion_phj import rrf_fusion
from retrieve.RAG.bge_reanker import bge_rerank
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline
import pandas as pd

def rrf_fusion(bm25_results, faiss_results, k=60):
    fusion_scores = {}

    # BM25 结果，计算 RRF 分数
    for rank, (doc, idx) in enumerate(bm25_results):
        if idx not in fusion_scores:
            fusion_scores[idx] = 1 / (k + rank)
        else:
            fusion_scores[idx] += 1 / (k + rank)

    # FAISS 结果，计算 RRF 分数
    for rank, (doc, idx) in enumerate(faiss_results):
        if idx not in fusion_scores:
            fusion_scores[idx] = 1 / (k + rank)
        else:
            fusion_scores[idx] += 1 / (k + rank)

    # 根据融合分数对文档进行排序
    sorted_fusion_results = sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_fusion_results
######按照每行切分新闻 不添加噪声因素######
def load_and_process_txt(file_path):
    # 打开文件，并将每一行作为一个单独的新闻进行处理
    with open(file_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []
    
    # 对每行新闻单独进行分割
    for line in lines:
        if line.strip():  # 跳过空行
            split_texts = text_splitter.split_text(line.strip())  # 切分当前行
            for text in split_texts:
                # 将每个切分后的块包装成 Document 对象
                doc = Document(page_content=text)
                all_docs.append(doc)
    
    return all_docs

# FAISS 向量召回
def create_faiss_index(docs):
    embeddings_model = HuggingFaceEmbeddings(model_name='/home/extra1T/model_embeding/ai-modelscope/gte-large-zh')
    vector_store = FAISS.from_documents(docs, embeddings_model)
    return vector_store

#####直接调用langchain进行切分 会有一部分噪声和其他新闻进行混合######
def load_and_process_txt1(file_path):
    loader = TextLoader(file_path, encoding='utf8')
    documents = loader.load()
    
    # 分割文本，保持上下文一致性
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs
#加载bm25 和faiss 向量库 
def load_faiss_bm25(data_path,embedding_model_path,faiss_path,bm25_path):
     docs = load_and_process_txt(data_path)
     embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
     vector_store = load_faiss_index(faiss_path, embeddings_model)
     bm25 = load_bm25_model(bm25_path)
     return [docs,vector_store,bm25]
#批量处理问题
def load_questions(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 假设 CSV 文件有一列叫 "questions"
        for row in reader:
            questions.append(row['question'].strip())  # 读取并清理每一行的问题
    return questions
### 处理单个问题 
def load_questions111(question_str):
    questions = []
    questions.append(question_str.strip())  # 读取并清理字符串的问题
    return questions
# def Search(query,embedding_model_path,faiss_path,bm25_path):
#     embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
#     vector_store = load_faiss_index(faiss_path, embeddings_model)
#     bm25 = load_bm25_model(bm25_path)
#     questions = load_questions111(query)
#     process_questions_with_progress(questions, docs, bm25, vector_store)
####加载 bm25
def load_bm25_model(file_path):
    with open(file_path, 'rb') as f:
        bm25 = pickle.load(f)
    print(f"BM25 模型已从 {file_path} 加载")
    return bm25

##保存bm25
def save_bm25_model(bm25, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 模型已保存到 {file_path}")

#建立faiss 向量库 和bm25向量库 并且保存 
def create_faiss_bm25(file_path,save_faiss_path,save_bm25_model_file):
    docs = load_and_process_txt(file_path)
    vector_store = create_faiss_index(docs)
    save_faiss_index(vector_store,save_faiss_path)
    tokenized_docs = process_for_bm25(docs)
    bm25 = BM25Okapi(tokenized_docs)
    save_bm25_model(bm25, save_bm25_model_file)
    return [docs,vector_store,bm25]
#建立faiss 向量库 并且 根据指定问题进行查询
def create_fa(query,file_path,topk,path):
    docs = load_and_process_txt1(file_path)
    vector_store = create_faiss_index(docs)
    save_faiss_index(vector_store,path)
    result_with_score = vector_store.similarity_search_with_score(query, k=topk)
    faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]  # 直接访问 page_content
    return faiss_topk_matches
#保存faiss
def save_faiss_index(vector_store, save_path):
    # 保存向量库和对应的文档到指定路径
    vector_store.save_local(save_path)
    print(f"FAISS 向量库已保存到 {save_path}")

##加载faiss
def load_faiss_index(load_path, embeddings_model):
    # 从指定路径加载向量库
    vector_store = FAISS.load_local(load_path, embeddings_model)
    print(f"FAISS 向量库已从 {load_path} 加载")
    return vector_store

# 分词处理
def process_for_bm25(docs):
    tokenized_docs = [jieba.lcut(doc.page_content) for doc in docs]
    return tokenized_docs


def process_questions(questions, docs, bm25, vector_store, topk=2):
    # final_contexts = []  # 用于保存所有问题的上下文
    query_tokens = jieba.lcut(questions)

    # BM25 召回
    bm25_doc_scores = bm25.get_scores(query_tokens)
    # bm25_topk_indices = bm25_doc_scores.argsort()[::-1]
    bm25_topk_indices = bm25_doc_scores.argsort()[-10:][::-1]
    bm25_topk_matches = [(docs[idx].page_content, idx) for idx in bm25_topk_indices]

    # FAISS 召回
    result_with_score = vector_store.similarity_search_with_score(questions, k=10)
    faiss_topk_matches = [(res.page_content, docs.index(res)) for res, score in result_with_score]

    # RRF 融合
    rrf_results = rrf_fusion(bm25_topk_matches, faiss_topk_matches, k=60)

    # 取融合后的 Top 3
    top3_docs = [docs[idx].page_content for idx, _ in rrf_results[:10]]

    # 使用 BGE 模型进行重排序
    reranked_results = bge_rerank(questions, top3_docs)

    # 获取重排序后的 Top 3 文档内容
    topk_fused_docs = [doc[0] for doc in reranked_results[:topk]]

    # 将 Top 3 的文档内容拼接成上下文
    context = "\n\n".join(topk_fused_docs)

    # 将每个问题和对应的上下文信息保存
    # final_contexts.append({"query": query, "context": context})

    return context

def process_questions_with_progress(questions, docs, bm25, vector_store):
    final_contexts = []
    
    # 使用 tqdm 显示处理进度
    for question in tqdm(questions, desc="Processing Questions"):
        # 对每个问题进行多路召回和重排序
        context = process_questions(question, docs, bm25, vector_store)
        
        # 将问题和上下文存入 final_contexts 列表中
        final_contexts.append({
            'query': question,
            'context': context
        })
    
    return final_contexts
#####查看保存好的csv文件 
def count_rows_columns_pandas(file_path):
    df = pd.read_csv(file_path)
    return df.shape[0], df.shape[1]  # 返回行数和列数


######加载模型#############
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",offload_folder="offload",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

##删除生成的question列 在原来的csv表格上进行操作
def drop(file_path):
    df = pd.read_csv(file_path)
    if 'Question' in df.columns:
        df.drop('Question', axis=1, inplace=True)
    df.to_csv(file_path, index=False)
    print("已删除 'question' 列，并保存了更改。")

#读取json文件
def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        final_contexts = json.load(file)
    return final_contexts

##保存json文件
def save_json(json_file_path,final_contexts):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_contexts, f, ensure_ascii=False, indent=4)

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
必须根据检索到的内容进行回答，不要根据你自身的知识储备进行回答，如果检索到的内容中不包含问题对应的知识点，请务必直接回答“结合给定的资料无法回答这个问题。”这句话,同时给出我无法回答用户问题的理由。

检索到的上下文：{context}

用户问题：{question}
回答："""