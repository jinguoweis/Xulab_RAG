from loguru import logger
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
import torch
from transformers import pipeline
import pandas as pd
from retrieve.RAG.utils_process import *
class Searcher:
    def __init__(self, data_path,embedding_model_path,faiss_path,bm25_path):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
        self.vector_store = load_faiss_index(faiss_path, self.embeddings_model)
        logger.info("load vec_model done")
        self.bm25 = load_bm25_model(bm25_path)
        logger.info("load bm25_model done")
        self.docs = load_and_process_txt(data_path)
        logger.info("load docs done")
    # def __init__(self, data_path,embedding_model_path,faiss_path,bm25_path):
    #     self.docs = load_and_process_txt(data_path)
    #     logger.info("load docs done")
    #     self.embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_path)
    #     self.vector_store = create_faiss_index(self.docs)
    #     logger.info("load vec_model done")
    #     save_faiss_index(self.vector_store,faiss_path)
    #     self.tokenized_docs = process_for_bm25(self.docs)
    #     self.bm25 = BM25Okapi(self.tokenized_docs)
    #     save_bm25_model(self.bm25,bm25_path)
    #     logger.info("load bm25_model done")
    def search(self, query):
        logger.info("request: {}".format(query))
        questions = load_questions111(query)
        answer = process_questions_with_progress(questions, self.docs, self.bm25, self.vector_store)
        logger.info("response: {}".format(answer))
        return answer
