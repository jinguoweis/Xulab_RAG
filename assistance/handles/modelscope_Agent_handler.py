# coding=utf-8
# Filename:    llm_handler.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型服务handler

import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
from assistance.models.modelscope_Agent_model import Agentmodel

class Agent_LlmHandler(RequestHandler):
    
    def initialize(self, llm_model:Agentmodel):
        self.llm_model = llm_model
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }
        
        # response_body = {
        #     "answer":"XXX"
        # }

    # 定义了一个异步的POST处理函数，它会处理HTTP POST请求。在Tornado中，post方法默认处理POST请求。
    async def post(self):
        logger.info("request: {}".format(json_decode(self.request.body).get("query", "")))
        # bs_agent = BSAgentExecutor(self.llm_model)
        # answer = bs_agent.run(json_decode(self.request.body).get("query", ""),print_info=True)
        # logger.info("dsadsa: {}".format("开始预测..."))
        # logger.info("dsadsdsadasa: {}".format(self.llm_model))
        answer = self.llm_model.generate(json_decode(self.request.body).get("query", ""))
        response_body = {"answer": answer}
        logger.info("response: {}".format(response_body))

        # 将响应体发送给客户端。
        self.write(response_body)
def Agent_StartLlmHandler(request_config: dict, llm_model: Agentmodel):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], Agent_LlmHandler, {"llm_model":llm_model})]
    # 创建Tornado应用程序实例，传入路由。
    app = Application(handlers=handler_routes)
    #  创建HTTP服务器实例
    http_server = HTTPServer(app)
    # 使服务器监听配置中指定的端口
    http_server.listen(request_config["port"])
    # 启动事件循环，这将持续监听和处理请求直到程序被关闭
    tornado.ioloop.IOLoop.current().start()