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
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import PromptTemplate
import json
from langchain.docstore.document import Document
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils_process import *
import argparse
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 导入其他模块和包
# from your_module import YourClass
# from another_module import another_function

def main(args):

    logging.info("Reinforced Agent Generator initialization...")
        # 使用命令行参数
    embedding_model_path = args.embedding_model
    bge_model_path = args.bge_model
    data_path = args.data_path
    faiss_path = args.faiss_path
    bm25_path = args.bm25_path
    model_path = args.model_path
    questions_file_path = args.test_question_path
    save_json_path = args.save_context_path
    load_context_path = args.load_context_path
    save_answer_path = args.save_answer_path
    save_faiss_path = args.save_faiss_path
    save_bm25_path = args.save_bm25_path
    ##打印输出
    logging.info(f"使用嵌入模型: {embedding_model_path}")
    logging.info(f"使用重排序模型: {bge_model_path}")
    logging.info(f"加载数据集: {data_path}")
    logging.info(f"加载faiss向量模型: {faiss_path}")
    logging.info(f"加载bm25模型: {bm25_path}")
    logging.info(f"加载上下文: { save_json_path}")
    #有保存好的 faiss向量库索引和bm25索引 就直接加载 并且传入原始数据路径 生成docs切片 此下三行代码适用于没有保存提取到的上下文
    # docs,vector_store,bm25 = load_faiss_bm25(data_path,embedding_model_path,faiss_path,bm25_path)
    # questions = load_questions(questions_file_path)
    # final_contexts = process_questions_with_progress(questions, docs, bm25, vector_store)
    # save_json(save_json_path,final_contexts)


    ### 如果没有上下文并且没有保存好的faiss索引和bm25索引 用如下三行代码进行处理 有的话就直接加载 在这里我们已经保存过了 所以直接加载
    # docs,vector_store,bm25 = create_faiss_bm25(data_path,save_faiss_path,save_bm25_path)
    # final_contexts = process_questions_with_progress(questions, docs, bm25, vector_store)
    # save_json(save_json_path,final_contexts)

    final_contexts = load_json(load_context_path)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = load_model(model_path)
    rag_chain = LLMChain(
    llm=llm,
    prompt=prompt
)
    # 准备输出文件
    header = ['Question', 'Answer']
# 存储问题和答案到 CSV 文件
    with open(save_answer_path, 'w', encoding='utf-8', newline='') as csvfile:
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

    print(f"所有问题和答案已经写入到 {save_answer_path}")
    # 示例代码，实际代码应根据项目需求编写
    # model = YourClass()
    # model.train()
    # model.evaluate()
    drop(save_answer_path)
    logging.info("答案生成完毕。")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="基于RAG+LLM的检索系统研究")
    parser.add_argument('--embedding_model', type=str,help='嵌入模型', default='/home/extra1T/model_embeding/ai-modelscope/gte-large-zh', required=False)
    parser.add_argument('--bge_model', type=str, help='重排序模型', default='/home/extra1T/model_embeding/Xorbits/bge-reranker-base', required=False)
    parser.add_argument('--data_path', type=str, help='数据集', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt', required=False)
    parser.add_argument('--faiss_path', type=str, help='保存好的faiss向量模型', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_faiss', required=False)
    parser.add_argument('--bm25_path', type=str, help='保存好的bm25模型', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/model.pkl', required=False)
    parser.add_argument('--save_faiss_path', type=str, help='创建faiss向量模型', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_faiss', required=False)
    parser.add_argument('--save_bm25_path', type=str, help='创建bm25模型', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/model.pkl', required=False)
    parser.add_argument('--model_path', type=str, help='QWEN2-7B', default='/home/extra1T/model_embeding/qwen/Qwen2-7B', required=False)
    parser.add_argument('--test_question_path', type=str, help='测试问题', default='/home/extra1T/jin/GoMate/RAG/data/test_question.csv', required=False)
    parser.add_argument('--save_context_path', type=str, help='保存上下文', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/final_context.json', required=False)
    parser.add_argument('--load_context_path', type=str, help='加载上下文', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/final_contexts_question.json', required=False)
    parser.add_argument('--save_answer_path', type=str, help='保存答案', default='/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/answer.csv', required=False)
    # 添加其他需要的参数

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 调用主函数
    main(args)
