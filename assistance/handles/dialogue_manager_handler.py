from loguru import logger
from tqdm import tqdm
import json
import pickle
import jieba
import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
from assistance.console.dialogue_manager import BSAgentExecutor
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
from retrieve.RAG.utils_process import *
from client import run_client
from copy import deepcopy
from typing import Dict, List, Optional, Union
import re
import sys
# sys.path.append('/home/extra1T/jin/app')
# from model.llm import LLM
class DialogueManagerHandler(RequestHandler):
    def initialize(self, dialogue_manager:BSAgentExecutor):
        self.dialogue_manager = dialogue_manager
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":"XXX"
        # }

    async def post(self):
        logger.info('11111111111111111')
        answer = self.dialogue_manager.run(json_decode(self.request.body).get("query", ""))
        # logger.info(answer)
        response_body = {"answer": answer}
        
        logger.info("response: {}".format(response_body))
        # response_body = answer
        self.write(response_body)

def StartDialogueManagerHandler(request_config: dict, dialogue_manager: BSAgentExecutor):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], DialogueManagerHandler, {"dialogue_manager":dialogue_manager})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()