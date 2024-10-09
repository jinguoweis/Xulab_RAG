# coding=utf-8
# Filename:    main_service_online.py
# Author:      ZENGGUANRONG
# Date:        2023-09-10
# description: tornado服务启动核心脚本

import sys
from loguru import logger

import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
from multiprocessing import Process

import sys
sys.path.append('/home/extra1T/jin/GoMate/RAG')
from assistance.handles.search_handler import SearcherHandler,StartSearcherHandler
from assistance.searcher.search import Searcher
from assistance.models.chatglm3_model import LlmModel
from assistance.models.Qwen2_llm_model import QwenLlmModel
from assistance.handles.llm_handler import StartLlmHandler
from assistance.handles.Qwen2_llm_handler import Qwen2StartLlmHandler
from assistance.models.modelscope_Agent_model import Agentmodel
from assistance.handles.modelscope_Agent_handler import Agent_LlmHandler,Agent_StartLlmHandler
from assistance.console.dialogue_manager import BSAgentExecutor
from assistance.handles.dialogue_manager_handler import DialogueManagerHandler,StartDialogueManagerHandler
# from src.server.handlers.vec_model_handler import VecModelHandler,StartVecModelHandler
def launch_service(config, model_mode):
    if model_mode == "llm_model":
        # 解决windows下多进程使用pt会导致pickle序列化失败的问题，https://blog.csdn.net/qq_21774161/article/details/127145749
        llm_model = LlmModel(config["process_llm_model"]["model_path"], config["process_llm_model"]["model_config"])
        StartLlmHandler(config["process_llm_model"], llm_model)
        # processes = [process_llm_model]
        # for process in processes:
        #     process.start()
        # for process in processes:
        #     process.join()
    if model_mode == "qwen2_llm_model":
        llm_model = QwenLlmModel(config["process_Qwen2_llm_model"]["model_path"])
        Qwen2StartLlmHandler(config["process_Qwen2_llm_model"],llm_model)
    if model_mode == "searcher":
        searcher = Searcher(config["process_searcher"]["data_path"], config["process_searcher"]["embedding_model_path"],config["process_searcher"]["faiss_path"],config["process_searcher"]["bm25_path"])
        StartSearcherHandler(config["process_searcher"], searcher)
    if model_mode == "modelscope_Agent_model":
        # 解决windows下多进程使用pt会导致pickle序列化失败的问题，https://blog.csdn.net/qq_21774161/article/details/127145749
        llm_model = Agentmodel(config["process_modelscope_Agent_model"]["model_path"])  #原来的被我注释掉了
        # agent_model = BSAgentExecutor(llm_model)
        Agent_StartLlmHandler(config["process_modelscope_Agent_model"], llm_model)
    if model_mode == "dialogue_manager":
        dialogue_manager = BSAgentExecutor(config["process_dialogue_manager"])
        StartDialogueManagerHandler(config["process_dialogue_manager"], dialogue_manager)
    # elif model_mode == "searcher":
    #     searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"])
    #     process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))

    #     dialogue_manager = DialogueManager(config["process_dialogue_manager"])
    #     process_dialogue_manager = Process(target=StartDialogueManagerHandler, args=(config["process_dialogue_manager"], dialogue_manager))
    #     # vec_model = VectorizeModel(config["process_vec_model"]["VEC_MODEL_PATH"])
    #     # process_vec_model = Process(target=StartVecModelHandler, args=(config["process_vec_model"], vec_model))

    #     # processes = [process_searcher]
    #     processes = [process_searcher, process_dialogue_manager]
    #     for process in processes:
    #         process.start()
    #     for process in processes:
    #         process.join()
    else:
        logger.info("init service error")


if __name__ == "__main__":
    config = {"process_searcher":{"port":9091, 
                                      "url_suffix":"/searcher", 
                                      "data_path":"/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/corpus copy.txt",
                                      "embedding_model_path":"/home/extra1T/model_embeding/ai-modelscope/gte-large-zh",
                                      "faiss_path":"/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_faiss",
                                      "bm25_path":"/home/extra1T/jin/GoMate/RAG/retrieve/RAG/save_bm25/model.pkl"},
             "process_llm_model":{"port":9092, 
                                      "url_suffix":"/llm_model", 
                                      "model_path":"/home/extra1T/model_embeding/chatglm3-6b",
                                      "model_config":{}},
             "process_Qwen2_llm_model":{"port":9093, 
                                      "url_suffix":"/qwen2_llm_model", 
                                      "model_path":"/home/extra1T/model_embeding/qwen/Qwen2-7B"},
            #  "process_dialogue_manager":{"port":9093, 
            #                           "url_suffix":"/dialogue_manager",
            #                           "config":{"search_url":"http://127.0.0.1:9090/searcher",
            #                                     "llm_url":"http://127.0.0.1:9092/llm_model"}},
             "process_modelscope_Agent_model":{"port":9094, 
                                      "url_suffix":"/modelscope_Agent_model", 
                                      "model_path":"/home/extra1T/model_embeding/damo/ModelScope-Agent-7B"},
             "process_dialogue_manager":{"port":9095, 
                                      "url_suffix":"/dialogue_manager",
                                      "config":{"modelscope_url":"http://127.0.0.1:9094/modelscope_Agent_model",
                                                "llm_url":"http://127.0.0.1:9092//llm_model",
                                                "search_url":"http://127.0.0.1:9091/searcher",
                                                "qwen2_llm_url":"http://127.0.0.1:9093/qwen2_llm_model"}}
    }
    launch_service(config, sys.argv[1])