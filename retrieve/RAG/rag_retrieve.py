from typing import Dict, DefaultDict,  List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import re
import json
import glob
# from utils.file_processor import read_jsonl, write_jsonl
from config import PDF_TXT_PATH
from retrieve.RAG.utils_process import *


class DocRetrieve:
    def __init__(self,
                 top_k: int = 15,
                 txt_folder: str = PDF_TXT_PATH
    ):
        self.top_k = top_k
        self.txt_folder = txt_folder
        self.doc_dict = dict()
        self.company_list = []

    def get_single_extra_knowledge(self, company, query):
        data_path = '/home/extra1T/jin/GoMate/RAG/data/corpus.txt'
        embedding_model_path = '/home/extra1T/model_embeding/ai-modelscope/gte-large-zh'
        faiss_path = '/home/extra1T/jin/GoMate/RAG/faiss_index'
        bm25_path = '/home/extra1T/jin/GoMate/faiss_index/bm25_model.pkl'
        docs,vector_store,bm25 = load_faiss_bm25(data_path,embedding_model_path,faiss_path,bm25_path)
        final_contexts = process_questions_with_progress([query], docs, bm25, vector_store)
        return final_contexts[0]['context']