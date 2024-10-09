# coding=utf-8
# Filename:    main_service_online.py
# Author:      ZENGGUANRONG
# Date:        2023-09-10
# description: tornado服务启动核心脚本

import sys
sys.path.append('/home/extra1T/jingw/FinQwen-main/solutions/3_hxjj/app/basic_rag')
from loguru import logger
import pandas as pd
import faiss,copy
import uuid
from PIL import Image
import numpy as np
import tempfile
import tornado.ioloop
from typing import Dict, Tuple
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
from multiprocessing import Process
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import sys
from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
# from server.handlers.search_handler import StartSearcherHandler
sys.path.append('/home/extra1T/jingw/FinQwen-main/solutions/3_hxjj/app/')
# from basic_rag.src.searcher.searcher import Searcher
# from basic_rag.src.models.vec_model.vec_model import VectorizeModel
# from basic_rag.src.models.llm.llm_model import LlmModel
# from basic_rag.src.dm.dialogue_manager import DialogueManager

# from basic_rag.src.server.handlers.search_handler import SearcherHandler,StartSearcherHandler
# # from src.server.handlers.vec_model_handler import VecModelHandler,StartVecModelHandler
# from basic_rag.src.server.handlers.llm_handler import LlmModel,StartLlmHandler
# from server.handlers.dialogue_manager_handler import DialogueManagerHandler, StartDialogueManagerHandler
sys.path.append('/home/workspace/NLP/jinguowei2/app/agent')
# from agent.bs_agent import BSAgentExecutor
# from agent.bs_agent import BSAgentExecutor
import os
from typing import List, Optional

import json
import requests
from pydantic import BaseModel, ValidationError
from requests.exceptions import RequestException, Timeout
import re
# BS_TASK_INSTRUCTION_TEMPLATE1 = """当前对话可以使用的插件信息如下，请一定调用当前插件信息来解决当前用户问题。调用插件则需要将插件调用请求按照json格式给出，必须包含api_name、url、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复；若无需调用插件，则直接给出对应回复即可：
# {tool_list}"""
LLM_MODEL_DICT = {
    "qwen_bs": {
        "type": "qwen",
        "name": "ModelScope-Agent-7B",
        "model_path": "/home/workspace/NLP/jinguowei2/ModelScope-Agent-7B",
        # "lora_path": "models/ckpt_agent",
        # "lora_2_path": "models/ckpt_sql"
    },
    
}
class Agentmodel:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True,torch_dtype=torch.float16)
        self.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
        # 加载原模型
        # if cfg.get("lora_path") is not None:
        #     self.model = PeftModel.from_pretrained(self.model, cfg["lora_path"], adapter_name='agent_lora')
        # if cfg.get("lora_2_path") is not None:
        #     self.model.load_adapter(cfg["lora_2_path"], adapter_name='sql_lora')
        
    
    def original_generate(self, prompt, history=None):
        # with self.model.disable_adapter():
        #使用上下文管理器（context manager）禁用模型的适配器。适配器是用来增强模型功能的插件，通过禁用适配器，可以确保模型仅使用原始的GPT-3.5功能。
        response, _ = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
        return response

    def generate(self,prompt,history=None):
        # self.model.set_adapter('agent_lora')
        response, _ = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
        return response
        
    def sql_generate(self,prompt,history=None):
        # self.model.set_adapter('sql_lora')
        response, history = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
        return response,history
    # def sql_generate(self,prompt,history=None):
    #     # self.model.set_adapter('sql_lora')
    #     response, history = self.model.chat(self.tokenizer, prompt, history=history, generation_config=self.generation_config)
    #     return response,history
model_path = '/home/workspace/NLP/jinguowei2/ModelScope-Agent-7B'
def get_llm_cls(llm_type):
    if llm_type == 'qwen':
        return Agentmodel(model_path)
    # elif llm_type == 'baichuan':
    #     return Baichuan
    # elif llm_type == 'chatglm':
    #     return ChatGLM
    # elif llm_type == 'test':
    #     return TestLM
    else:
        raise ValueError(f'Invalid llm_type {llm_type}')


class LLMFactory:
    @staticmethod
    def build_llm(model_name):
        cfg = LLM_MODEL_DICT.get(model_name, {'type': 'test'})
        # cfg.update(additional_cfg)
        llm_type = cfg.pop('type')
        llm_cls = get_llm_cls(llm_type)
        # llm_cfg = cfg
        return llm_cls
model_name = 'qwen_bs'

llm = LLMFactory.build_llm(model_name)
BS_CHAT_KNOWLEDGE_TEMPLATE="""------检索内容开始------
{extra_knowledge}
------检索内容结束------

用户问题：{user_question}。
完全根据检索内容结合问题回答用户问题，将问题和答案结合后输出。注意不要输出“根据检索”。
"""
BS_TASK_INSTRUCTION_TEMPLATE1 = """当前任务可以使用的插件信息如下，请尽可能地调用插件来解决当前用户问题，将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
实在没办法了才直接执行任务。
{tool_list}"""
class OutputWrapper:
    """
    Wrapper for output of tool execution when output is image, video, audio, etc.
    In this wrapper, __repr__() is implemented to return the str representation of the output for llm.
    Each wrapper have below attributes:
        path: the path where the output is stored
        raw_data: the raw data, e.g. image, video, audio, etc. In remote mode, it should be None
    """

    def __init__(self) -> None:
        self._repr = None
        self._path = None
        self._raw_data = None

        self.root_path = os.environ.get('OUTPUT_FILE_DIRECTORY', None)
        if self.root_path and not os.path.exists(self.root_path):
            try:
                os.makedirs(self.root_path)
            except Exception:
                self.root_path = None

    def get_remote_file(self, remote_path, suffix):
        try:
            response = requests.get(remote_path)
            obj = response.content
            directory = tempfile.mkdtemp(dir=self.root_path)
            path = os.path.join(directory, str(uuid.uuid4()) + f'.{suffix}')
            with open(path, 'wb') as f:
                f.write(obj)
            return path
        except RequestException:
            return remote_path

    def __repr__(self) -> str:
        return self._repr

    @property
    def path(self):
        return self._path

    @property
    def raw_data(self):
        return self._raw_data


class ImageWrapper(OutputWrapper):
    """
    Image wrapper, raw_data is a PIL.Image
    """

    def __init__(self, image) -> None:

        super().__init__()

        if isinstance(image, str):
            if os.path.isfile(image):
                self._path = image
            else:
                self._path = self.get_remote_file(image, 'png')
            try:
                image = Image.open(self._path)
                self._raw_data = image
            except FileNotFoundError:
                # Image store in remote server when use remote mode
                raise FileNotFoundError(f'Invalid path: {image}')
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype(np.uint8))
                self._raw_data = image
            else:
                self._raw_data = image
            directory = tempfile.mkdtemp(dir=self.root_path)
            self._path = os.path.join(directory, str(uuid.uuid4()) + '.png')
            self._raw_data.save(self._path)

        self._repr = f'![IMAGEGEN]({self._path})'



def get_raw_output(exec_result: Dict):
    # get rwa data of exec_result
    res = {}
    for k, v in exec_result.items():
        if isinstance(v, OutputWrapper):
            # In remote mode, raw data maybe None
            res[k] = v.raw_data or str(v)
        else:
            res[k] = v
    return res


def display(llm_result: str, exec_result: Dict, idx: int):
    """Display the result of each round in jupyter notebook.
    The multi-modal data will be extracted.

    Args:
        llm_result (str): llm result
        exec_result (Dict): exec result
        idx (int): current round
    """
    from IPython.display import display, Pretty, Image, Audio, JSON
    idx_info = '*' * 50 + f'round {idx}' + '*' * 50
    display(Pretty(idx_info))

    match_action = re.search(
        r'<\|startofthink\|>```JSON([\s\S]*)```<\|endofthink\|>', llm_result)
    if match_action:
        result = match_action.group(1)
        try:
            json_content = json.loads(result, strict=False)
            display(JSON(json_content))
            llm_result = llm_result.replace(match_action.group(0), '')
        except Exception:
            pass

    display(Pretty(llm_result))

    exec_result = exec_result.get('result', '')

    if isinstance(exec_result, dict):
        display(JSON(exec_result))
    else:
        display(Pretty(exec_result))

    return
class PromptGenerator:

    def __init__(self,
                 plan_template: str = '',
                 task_template: str = '',
                 task_instruction_template: str = '',
                 user_template: str = '',
                 current_task_template: str = '',
                 sep='\n\n',
                 prompt_max_length: int = 10000):
        """
        prompt genertor
        Args:
            system_template (str, optional): System template, normally the role of LLM.
            instruction_template (str, optional): Indicate the instruction for LLM.
            user_template (str, optional): Prefix before user input. Defaults to ''.
            exec_template (str, optional): A wrapper str for exec result.
            assistant_template (str, optional): Prefix before assistant response.
            Some LLM need to manully concat this prefix before generation.
            prompt_max_length (int, optional): max length of prompt. Defaults to 2799.

        """

        self.plan_template = plan_template
        self.task_template = task_template
        self.task_instruction_template = task_instruction_template
        self.user_template = user_template
        self.current_task_template = current_task_template
        self.sep = sep

        self.prompt_max_length = prompt_max_length
        self.reset()

    def reset(self):
        self.prompt = ''

    def init_plan_prompt(self, user_question):
        """
        in this function, the prompt will be initialized.
        """
        self.system_prompt = self.plan_template
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        self.current_task_prompt = None
        self.task_result_prompt = None


    def init_task_prompt(self,user_question, tool_list):
        self.system_prompt = self.task_template + self.task_instruction_template.replace("{tool_list}",self.get_tool_str(tool_list))
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        self.current_task_prompt = self.current_task_template
        self.task_result_prompt = None
    def init_task_prompt111(self,user_question, tool_list):
        self.system_prompt1 = self.task_template + BS_TASK_INSTRUCTION_TEMPLATE1.replace("{tool_list}",self.get_tool_str(tool_list))
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        self.current_task_prompt = self.current_task_template
        self.task_result_prompt = None
    def generate(self):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        pass

    def get_tool_str(self, tool_list):
        """generate tool list string

        Args:
            tool_list (List[str]): list of tools

        """
        tool_str = self.sep.join(
            [f'{i+1}. {t}' for i, t in enumerate(tool_list)])
        return tool_str

    def get_history_str(self):
        """generate history string

        """
        history_str = ''
        for i in range(len(self.history)):
            history_item = self.history[len(self.history) - i - 1]
            text = history_item['content']
            if len(history_str) + len(text) + len(
                    self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str
BS_PLAN_DEFAULT_PROMPT = "你是一名高级智能助手，你可以先对问题进行分类，问题类型只有公司招股书咨询和股票基金数据查询两类，然后根据所给的信息列出回答该问题的任务列表。股票基金数据查询提供的表如下：A股公司行业划分表, A股票日行情表, 基金份额持有人结构, 基金债券持仓明细, 基金可转债持仓明细, 基金基本信息, 基金日行情表, 基金股票持仓明细, 基金规模变动表, 港股票日行情表。"
BS_TASK_DEFAULT_PROMPT = "你是一名高级智能助手，你需要根据当前提供的信息，执行当前任务。"
BS_CHAIN_PROMPT = "你是一名高级智能助手, 你需要针对用户问题，选择使用合适的插件。"
# BS_TASK_INSTRUCTION_TEMPLATE = """当前任务可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
# 若无需调用插件，直接执行任务，结果无需标志。
# BS_TASK_INSTRUCTION_TEMPLATE = """当前任务可以使用的插件信息如下，请一定调用插件，将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。
# {tool_list}"""
# BS_TASK_INSTRUCTION_TEMPLATE = """当前任务可以使用的插件信息如下，请尽可能地调用插件来解决当前用户问题，将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
# 实在没办法了才直接执行任务。
# {tool_list}"""
BS_TASK_INSTRUCTION_TEMPLATE = """当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、url、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复；若无需调用插件，则直接给出对应回复即可：
{tool_list}"""

SCHEME_STRUCTURE_DICT = {
'A股公司行业划分表': 
'''
字段 类型
股票代码 TEXT 
交易日期 TEXT
行业划分标准 TEXT
一级行业名称 TEXT
二级行业名称 TEXT
''',
'A股票日行情表': 
'''
字段 类型
股票代码 TEXT
交易日 TEXT
[昨收盘(元)] REAL
[今开盘(元)] REAL
[最高价(元)] REAL
[最低价(元)] REAL
[收盘价(元)] REAL
[成交量(股)] REAL
[成交金额(元)] REAL
''',
'基金份额持有人结构':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
公告日期 TIMESTAMP
截止日期 TIMESTAMP
机构投资者持有的基金份额 REAL
机构投资者持有的基金份额占总份额比例 REAL
个人投资者持有的基金份额 REAL
个人投资者持有的基金份额占总份额比例 REAL
定期报告所属年度 INTEGER
报告类型 TEXT
''',
'基金债券持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
债券类型 TEXT
债券名称 TEXT
持债数量 REAL
持债市值 REAL
持债市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型TEXT TEXT
''',
'基金可转债持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
对应股票代码 TEXT
债券名称 TEXT
数量 REAL
市值 REAL
市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型 TEXT
''',
'基金基本信息':
'''
字段 类型
基金代码 TEXT
基金全称 TEXT
基金简称 TEXT
管理人 TEXT
托管人 TEXT
基金类型 TEXT
成立日期 TEXT
到期日期 TEXT
管理费率 TEXT
托管费率 TEXT
''',
'基金日行情表':
'''
字段 类型
基金代码 TEXT
交易日期 TEXT
单位净值 REAL
复权单位净值 REAL
累计单位净值 REAL
资产净值 REAL
''',
'基金股票持仓明细':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
持仓日期 TEXT
股票代码 TEXT
股票名称 TEXT
数量 REAL
市值 REAL
市值占基金资产净值比 REAL
第N大重仓股 INTEGER
所在证券市场 TEXT
[所属国家(地区)] TEXT
报告类型 TEXT
''',
'基金规模变动表':
'''
字段 类型
基金代码 TEXT
基金简称 TEXT
公告日期 TIMESTAMP
截止日期 TIMESTAMP
报告期期初基金总份额 REAL
报告期基金总申购份额 REAL
报告期基金总赎回份额 REAL
报告期期末基金总份额 REAL
定期报告所属年度 INTEGER
报告类型 TEXT
''',
'港股票日行情表':
'''
字段 类型
股票代码 TEXT
交易日 TEXT
[昨收盘(元)] REAL
[今开盘(元)] REAL
[最高价(元)] REAL
[最低价(元)] REAL
[收盘价(元)] REAL
[成交量(股)] REAL
[成交金额(元)] REAL
'''
}

BS_USER_QUESTION_TEMPLATE = "用户问题：{user_question}"
BS_CURRENT_TASK_TEMPLATE = "当前任务：{current_task}"

BS_CHAT_KNOWLEDGE_TEMPLATE="""------检索内容开始------
{extra_knowledge}
------检索内容结束------

用户问题：{user_question}。
完全根据检索内容结合问题回答用户问题，将问题和答案结合后输出。注意不要输出“根据检索”。
"""

# BS_CHAT_KNOWLEDGE_TEMPLATE="""------检索内容开始------
# {extra_knowledge}
# ------检索内容结束------

# 用户问题：{user_question}。
# 完全根据检索内容结合问题回答用户问题，将问题和答案结合后输出；
# 若在检索内容中无答案，输出“问题” + “并未在招股意向书中详细说明”，如用户问题：“上海华铭智能终端设备股份有限公司的首发战略配售结果如何？”，输出：“上海华铭智能终端设备股份有限公司的首发战略配售具体情况并未在招股意向书中详细说明。”。"""

BS_SQL_GENERATOR_TEMPLATE="""你是一名高级数据库工程师，请你根据所提供的表结构说明以及用户问题，生成sql语句，数据库为sqlite，你生成的sql语句格式必须符合sqlite格式。
------表结构说明开始------
{table_structure_introduction}
------表结构说明结束------

用户问题：{user_question}。
注意：答案只需要sql语句，不需要其他任何输出。
"""

BS_SQL_GENERATOR_TEMPLATE_1="你是一名sqlite数据库开发人员，精通sql语言，你需要根据已知的10张表的表名、字段名和用户输入的问题编写sql\n\n" \
             "{'表名': '基金基本信息', '字段名': ['基金代码', '基金全称', '基金简称', '管理人', '托管人', '基金类型', '成立日期', '到期日期', '管理费率', '托管费率']}\n" \
             "{'表名': '基金股票持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '股票代码', '股票名称', '数量', '市值', '市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金债券持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '债券类型', '债券名称', '持债数量', '持债市值', '持债市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金可转债持仓明细', '字段名': ['基金代码', '基金简称', '持仓日期', '对应股票代码', '债券名称', '数量', '市值', '市值占基金资产净值比', '第N大重仓股', '所在证券市场', '[所属国家(地区)]', '报告类型']}\n" \
             "{'表名': '基金日行情表', '字段名': ['基金代码', '交易日期', '单位净值', '复权单位净值', '累计单位净值', '资产净值']}\n" \
             "{'表名': 'A股票日行情表', '字段名': ['股票代码', '交易日', '[昨收盘(元)]', '[今开盘(元)]', '[最高价(元)]', '[最低价(元)]', '[收盘价(元)]', '[成交量(股)]', '[成交金额(元)]']}\n" \
             "{'表名': '港股票日行情表', '字段名': ['股票代码', '交易日', '[昨收盘(元)]', '[今开盘(元)]', '[最高价(元)]', '[最低价(元)]', '[收盘价(元)]', '[成交量(股)]', '[成交金额(元)]']}\n" \
             "{'表名': 'A股公司行业划分表', '字段名': ['股票代码', '交易日期', '行业划分标准', '一级行业名称', '二级行业名称']}\n" \
             "{'表名': '基金规模变动表', '字段名': ['基金代码', '基金简称', '公告日期', '截止日期', '报告期期初基金总份额', '报告期基金总申购份额', '报告期基金总赎回份额', '报告期期末基金总份额', '定期报告所属年度', '报告类型']}\n" \
             "{'表名': '基金份额持有人结构', '字段名': ['基金代码', '基金简称', '公告日期', '截止日期', '机构投资者持有的基金份额', '机构投资者持有的基金份额占总份额比例', '个人投资者持有的基金份额', '个人投资者持有的基金份额占总份额比例', '定期报告所属年度', '报告类型']}\n\n" \
             "请根据以下用户输入编写sql。\n用户输入: {user_question}"

BS_CHAT_SQLRESULT_TEMPLATE="""问题：“{user_question}”。
答案：“{sql_result}”。

将问题的内容和答案的内容融合的文字内容输出。注意不要输出“问题：”或“答案：”。
"""

class BSPromptGenerator(PromptGenerator):
    def __init__(self,
                 plan_template=BS_PLAN_DEFAULT_PROMPT,
                 task_template=BS_TASK_DEFAULT_PROMPT,
                 task_instruction_template=BS_TASK_INSTRUCTION_TEMPLATE,
                 user_template=BS_USER_QUESTION_TEMPLATE,
                 current_task_template=BS_CURRENT_TASK_TEMPLATE,
                 sep='\n\n',
                 prompt_max_length=10000):
        super().__init__(plan_template, task_template, task_instruction_template, user_template, current_task_template, sep,
                         prompt_max_length)


    def generate(self, task_no=None):
        # init plan
        if task_no is None:
            prompt_list = [self.system_prompt,
                           self.user_prompt]
        # execute tasks
        else:
            # no task result
            if not self.task_result_prompt:
                prompt_list = [self.system_prompt,
                               self.user_prompt,
                               self.current_task_prompt]
            else:
                prompt_list = [self.system_prompt,
                               self.task_result_prompt,
                               self.user_prompt,
                               self.current_task_prompt]
        return self.sep.join(prompt_list)
    def generate1(self, task_no=None):
        # init plan
        if task_no is None:
            prompt_list = [self.system_prompt1,
                           self.user_prompt]
        # execute tasks
        else:
            # no task result
            if not self.task_result_prompt:
                prompt_list = [self.system_prompt1,
                               self.user_prompt,
                               self.current_task_prompt]
            else:
                prompt_list = [self.system_prompt1,
                               self.task_result_prompt,
                               self.user_prompt,
                               self.current_task_prompt]
        return self.sep.join(prompt_list)
    def update_task_prompt(self, current_task):
        self.current_task_prompt = self.current_task_template.replace("{current_task}", current_task)

class BSChainPromptGenerator(PromptGenerator):
    def __init__(self, 
                 chain_template=BS_CHAIN_PROMPT,
                 task_instruction_template=BS_TASK_INSTRUCTION_TEMPLATE,
                 user_template=BS_USER_QUESTION_TEMPLATE,
                 sep='\n\n'):
        self.chain_template = chain_template
        self.task_instruction_template = task_instruction_template
        self.user_template = user_template
        self.sep = sep
    
    def init_prompt(self, tool_list):
        self.system_prompt = self.chain_template + self.task_instruction_template.replace("{tool_list}",self.get_tool_str(tool_list))
        
        
    
    def generate(self, user_question):
        self.user_prompt = self.user_template.replace("{user_question}",user_question)
        return self.sep.join([self.system_prompt, self.user_prompt])
def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)
MODELSCOPE_API_TOKEN = os.getenv('MODELSCOPE_API_TOKEN')

MAX_RETRY_TIMES = 3


class ParametersSchema(BaseModel):
    name: str
    description: str
    required: Optional[bool] = True


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ParametersSchema]

class OutputParser:

    def parse_response(self, response):
        raise NotImplementedError

class BSOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if '<|startofthink|>' not in response or '<|endofthink|>' not in response:
            return None, None
        try:
            # use regular expression to get result
            re_pattern1 = re.compile(
                pattern=r'<\|startofthink\|>([\s\S]+)<\|endofthink\|>')
            think_content = re_pattern1.search(response).group(1)

            re_pattern2 = re.compile(r'{[\s\S]+}')
            think_content = re_pattern2.search(think_content).group()

            json_content = json.loads(think_content.replace('\n', '').replace('""','"'))
            action = json_content.get('api_name',
                                      json_content.get('name', 'unknown')).strip(' ')
            parameters = json_content.get('parameters', {})

            return action, parameters

        except Exception as e:
            return None, None


class QwenOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'Action' not in response or 'Action Input:' not in response:
            return None, None
        try:
            # use regular expression to get result
            re_pattern1 = re.compile(
                pattern=r'Action:([\s\S]+)Action Input:([\s\S]+)')
            res = re_pattern1.search(response)
            action = res.group(1).strip()
            action_para = res.group(2)

            parameters = json.loads(action_para.replace('\n', ''))

            print(response)
            print(action, parameters)
            return action, parameters
        except Exception:
            return None, None
class Tool:
    """
    a base class for tools.
    when you inherit this class and implement new tool, you should provide name, description
    and parameters of tool that conforms with schema.

    each tool may have two call method: _local_call(execute tool in your local environment)
    and _remote_call(construct a http request to remote server).
    corresponding to preprocess and postprocess method may need to be overrided to get correct result.
    """
    name: str = 'tool'
    description: str = 'This is a tool that ...'
    parameters: list = []

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        self.is_remote_tool = self.cfg.get('is_remote_tool', False)

        # remote call
        self.url = self.cfg.get('url', '')
        self.token = self.cfg.get('token', '')
        self.header = {
            'Authorization': self.token or f'Bearer {MODELSCOPE_API_TOKEN}'
        }

        try:
            all_para = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_para)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json(exclude_none=True)

    def __call__(self, *args, **kwargs):
        if self.is_remote_tool:
            return self._remote_call(*args, **kwargs)
        else:
            return self._local_call(*args, **kwargs)

    def _remote_call(self, *args, **kwargs):
        if self.url == '':
            raise ValueError(
                f"Could not use remote call for {self.name} since this tool doesn't have a remote endpoint"
            )

        remote_parsed_input = json.dumps(
            self._remote_parse_input(*args, **kwargs))

        origin_result = None
        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                response = requests.request(
                    'POST',
                    self.url,
                    headers=self.header,
                    data=remote_parsed_input)
                if response.status_code != requests.codes.ok:
                    response.raise_for_status()

                origin_result = json.loads(
                    response.content.decode('utf-8'))['Data']

                final_result = self._parse_output(origin_result, remote=True)
                return final_result
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def _local_call(self, *args, **kwargs):
        return

    def _remote_parse_input(self, *args, **kwargs):
        return kwargs

    def _local_parse_input(self, *args, **kwargs):
        return args, kwargs

    def _parse_output(self, origin_result, *args, **kwargs):
        return {'result': origin_result}

    def __str__(self):
        return self._str
    def get_function(self):
        return self._function

    def parse_pydantic_model_to_openai_function(self, all_para: dict):
        '''
        this method used to convert a pydantic model to openai function schema
        such that convert
        all_para = {
            'name': get_current_weather,
            'description': Get the current weather in a given location,
            'parameters': [{
                'name': 'image',
                'description': '用户输入的图片',
                'required': True
            }, {
                'name': 'text',
                'description': '用户输入的文本',
                'required': True
            }]
        }
        to
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "用户输入的图片",
                    },
                    "text": {
                        "type": "string",
                        "description": "用户输入的文本",
                    },
                "required": ["image", "text"],
            },
        }
        '''

        function = {
            'name': all_para['name'],
            'description': all_para['description'],
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        }
        for para in all_para['parameters']:
            function['parameters']['properties'][para['name']] = {
                'type': 'string',
                'description': para['description']
            }
            if para['required']:
                function['parameters']['required'].append(para['name'])

        return function
class AliyunRenewInstanceTool(Tool):
    description = '续费一台包年包月ECS实例'
    name = 'RenewInstance'
    parameters: list = [{
        'name': 'instance_id',
        'description': 'ECS实例ID',
        'required': True
    },
        {
            'name': 'period',
            'description': '续费时长以月为单位',
            'required': True
        }
    ]
    # def __call__(self, remote=False, *args, **kwargs):
    #     if self.is_remote_tool or remote:
    #         return self._remote_call(*args, **kwargs)
    #     else:
    #         return self._local_call(*args, **kwargs)

    # def _remote_call(self, *args, **kwargs):
    #     pass

    def _local_call(self, *args, **kwargs):
        instance_id = kwargs['instance_id']
        period = kwargs['period']
        # return {'result': f'成功为{instance_id}续费，续费时长{period}月'}
        return f'成功为{instance_id}续费，续费时长{period}月'
class AMAPPOIQuery(Tool):
    description = '获取对应地区地点的poi信息,你要把把我的地区地点名称连在一起输出作为一个参数，不能让他们之间用逗号分开。比如：问题是帮我查看杭州市的海底捞poi，你需要解析出来的参数是"杭州市海底捞"的参数，是连在一起的。'
    name = 'amap_poi_query'
    parameters: list = [
        {
            'name': 'keywords',
            'description': 'Text information for the location to be retrieved',
            'required': True
        }
    ]

    def __init__(self, cfg={}):
        # self.cfg = cfg.get(self.name, {})

        # remote call
        self.url = 'https://restapi.amap.com/v5/place/text?key={key}&keywords=%{keywords}'
        # self.token = self.cfg.get('token', os.environ.get('AMAP_TOKEN', ''))
        self.token = '451f1780d0d5ac1ac14ea48010d04f4d'
        assert self.token != '', 'weather api token must be acquired through ' \
            'https://lbs.amap.com/api/webservice/guide/create-project/get-key and set by AMAP_TOKEN'

        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)


    def __call__(self, *args, **kwargs):
        # result = []
        keywords = kwargs['keywords']
        response = requests.get(
            self.url.format(keywords=keywords, key=self.token))
        data = response.json()
        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            input = data.get('pois')
            # input = data
            # user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', args[0]).replace('{extra_knowledge}',input)
            # result = llm.original_generate(user_input)
            # return {'result': result}
            # return {'查询到的poi信息为:': input}
            return input
class VecIndex:
    def __init__(self) -> None:
        self.index = ""
    
    def build(self, index_dim):
        description = "HNSW64"
        measure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, measure)
    #  1   建立一个向量索引。它接受一个参数index_dim，表示向量的维度。在方法内部，它使用了HNSW（Hierarchical Navigable Small World）算法和L2距离度量（faiss.METRIC_L2）来创建一个具有指定维度的索引对象。
    
    def insert(self, vec):
        self.index.add(vec)
    #  2   向索引中插入单个向量。它接受一个参数vec，表示要插入的向量。
    def batch_insert(self, vecs):
        self.index.add(vecs)
    
    def load(self, read_path):
        # read_path: XXX.index
        self.index = faiss.read_index(read_path)
        ##444      读取 保存 的 self.index_folder_path + "/invert_index.faiss"

    def save(self, save_path):
        # save_path: XXX.index
        faiss.write_index(self.index, save_path)
    ##   3    此时的 self.index是插入过后的 faiss向量库 给保存下来了 self.index_folder_path + "/invert_index.faiss"
    
    def search(self, vec, num):
        # id, distance
        return self.index.search(vec, num)
    ##5    查 开始检索 在存入的 向量库中 
class VecSearcher:
    def __init__(self,classes):
        self.invert_index = VecIndex() # 检索倒排，使用的是索引是VecIndex
        self.forward_index = [] # 检索正排，实质上只是个list，通过ID获取对应的内容
        self.INDEX_FOLDER_PATH_TEMPLATE = "/home/workspace/NLP/jinguowei2/app/basic_rag/data/"+classes+"/{}"

    def build(self, index_dim, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)
        if not os.path.exists(self.index_folder_path) or not os.path.isdir(self.index_folder_path):
            os.mkdir(self.index_folder_path)

        self.invert_index = VecIndex()
        self.invert_index.build(index_dim) ##传入索引的维度

        self.forward_index = []
    
    def insert(self, vec, doc):
        self.invert_index.insert(vec)  ##将 处理好的 问题的embedding 的768 维度 传入 self.index中
        # self.invert_index.batch_insert(vecs)

        self.forward_index.append(doc) ## 存入的是 [ll["title"], ll] 形状的 东西  前面是标题
    
    def save(self):
        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
        #存入的是 ["人站在地球上为什么没有头朝下的感觉 ", {"qid": "qid_5982723620932473219", "category": "教育/科学-理工学科-地球科学", "title": "人站在地球上为什么没有头朝下的感觉 ", "desc": "", "answer": "地球上重力作用一直是指向球心的，因此\r\n只要头远离球心，人们就回感到头朝上。"}]
        ##格式的数据
        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")  ##存入的是 self.invert_index.insert(vec)  处理好的 问题的embedding 的768 维度 传入 self.index中
    
    def load(self, index_name):
        self.index_name = index_name
        self.index_folder_path = self.INDEX_FOLDER_PATH_TEMPLATE.format(index_name)

        self.invert_index = VecIndex()
        self.invert_index.load(self.index_folder_path + "/invert_index.faiss")

        self.forward_index = []
        with open(self.index_folder_path + "/forward_index.txt", encoding="utf8") as f:
            for line in f:
                self.forward_index.append(json.loads(line.strip()))   
                #self.forward_index 里面存储的数据变成下列格式了 self.forward_index[0]是 问题  self.forward_index[0][1]['answer']是答案
##['人站在地球上为什么没有头朝下的感觉 ',
 #{'qid': 'qid_5982723620932473219',
 # 'category': '教育/科学-理工学科-地球科学',
  #'title': '人站在地球上为什么没有头朝下的感觉 ',
  #'desc': '',
  #'answer': '地球上重力作用一直是指向球心的，因此\r\n只要头远离球心，人们就回感到头朝上。'}]
    
    def search(self, vecs, nums = 5):
        search_res = self.invert_index.search(vecs, nums)
        ##返回的 是 二维的元组 (array([[96.930145,...]],dtype=float32),array([[65611,73335,28723]]))  （1,3）形状 
        #后面的 是检索到的与目标句子最相似的向量库的索引编号 前面的 是这三个索引标号 查询到的库里面的向量 与目标句子(question)的计算的相似距离 越小越好  
        recall_list = []
        for idx in range(nums):
            # recall_list_idx, recall_list_detail, distance
            recall_list.append([search_res[1][0][idx], self.forward_index[search_res[1][0][idx]], search_res[0][0][idx]])
            #search_res[1][0][idx]对应上面的索引编号找的是 数组里的那一条语句   search_res[0][0][idx]] 存的是对应的距离   search_res[1][0][idx]   存的是库里的索引编号
        # recall_list = list(filter(lambda x: x[2] < 100, result))

        return recall_list
class SimcseModel(nn.Module):
    # https://blog.csdn.net/qq_44193969/article/details/126981581
    def __init__(self, pretrained_bert_path, pooling="cls") -> None:
        super(SimcseModel, self).__init__()

        self.pretrained_bert_path = pretrained_bert_path
        self.config = BertConfig.from_pretrained(self.pretrained_bert_path)
        
        self.model = BertModel.from_pretrained(self.pretrained_bert_path, config=self.config)
        self.model.eval()
        
        # self.model = None
        self.pooling = pooling
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.pooling == "cls":
            return out.last_hidden_state[:, 0]
        if self.pooling == "pooler":
            return out.pooler_output
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
class VectorizeModel:
    def __init__(self, ptm_model_path, device = "cpu") -> None:
        self.tokenizer = BertTokenizer.from_pretrained(ptm_model_path)
        self.model = SimcseModel(pretrained_bert_path=ptm_model_path, pooling="cls")
        self.model.eval()
        
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.DEVICE = device
        # logger.info(device)
        self.model.to(self.DEVICE)
        # logger.info("kkkkkkkkkkk")
        
        self.pdist = nn.PairwiseDistance(2)
    
    def predict_vec(self,query):
        q_id = self.tokenizer(query, max_length = 200, truncation=True, padding="max_length", return_tensors='pt')
        with torch.no_grad():
            q_id_input_ids = q_id["input_ids"].squeeze(1).to(self.DEVICE)
            q_id_attention_mask = q_id["attention_mask"].squeeze(1).to(self.DEVICE)
            q_id_token_type_ids = q_id["token_type_ids"].squeeze(1).to(self.DEVICE)
            q_id_pred = self.model(q_id_input_ids, q_id_attention_mask, q_id_token_type_ids)

        return q_id_pred
    #计算question 的嵌入向量

    def predict_vec_request(self, query):
        q_id_pred = self.predict_vec(query)
        return q_id_pred.cpu().numpy().tolist()
    
    def predict_sim(self, q1, q2):
        q1_v = self.predict_vec(q1)
        q2_v = self.predict_vec(q2)
        sim = F.cosine_similarity(q1_v[0], q2_v[0], dim=-1)
        return sim.numpy().tolist()  ##展开成 一维数组了 
class Searcher:
    def __init__(self, model_path, vec_search_path,classes):
        self.vec_model = VectorizeModel(model_path)
        logger.info("load vec_model done")

        self.vec_searcher = VecSearcher(classes)
        self.vec_searcher.load(vec_search_path)   ##vec_search_path是传入的 vec_index_test2023121301_20w 的文件夹的名字 调用 load函数 
        logger.info("load vec_searcher done")

    def rank(self, query, recall_result):
        rank_result = []
        for idx in range(len(recall_result)):
            new_sim = self.vec_model.predict_sim(query, recall_result[idx][1][0])#recall_result[idx][1][0]表示的是 在保存好的向量库与目标语句算出来的距离最相似的那个question
            #new_sim 是 计算得到的两个字符串之间的相似度 转换为NumPy数组，然后再转换为Python列表，并作为函数的返回值
            rank_item = copy.deepcopy(recall_result[idx])
            rank_item.append(new_sim) #rank_item是将相似度分别加在每一个 三位列表中 变成 四维列表 [73335,['哪些人切记不要吃花生?',{...}],191.32423,69] 69是计算出来的相似度
            rank_result.append(copy.deepcopy(rank_item)) #把新的这个四维列表保存起来 
        rank_result.sort(key=lambda x: x[3], reverse=True)
        #当 reverse 设为 True 时，列表将以降序排列（从大到小），而当 reverse 设为 False 时，列表将以升序排列（从小到大）。
        #对 rank_result 列表中的元素按照索引为3的位置上的值进行降序排序。 按照 相似度的值将列表进行降序排序 
        return rank_result # 返回降序排序的列表 
    
    def search(self, query):  #num =3 表示 topN 返回最接近的三个答案
        logger.info("request: {}".format(query))
        print('ss')
        q_vec = self.vec_model.predict_vec(query).cpu().numpy()
        #q_vec是 数据中每个问题的词向量表示 变成了 (1,768)的数组形状 不是tensor了
        # logger.info("q_vec: {}".format(q_vec))
        recall_result = self.vec_searcher.search(q_vec, 3)
        # recall_result的形状为 [## [65611,['人站在地球上为什么没有头朝下的感觉 ',
 #{'qid': 'qid_5982723620932473219',
 # 'category': '教育/科学-理工学科-地球科学',
  #'title': '人站在地球上为什么没有头朝下的感觉 ',
  #'desc': '',
  #'answer': '地球上重力作用一直是指向球心的，因此\r\n只要头远离球心，人们就回感到头朝上。'}],96.930145], [73335,[...],191.32423], ....] 的形状 
        # logger.info("recall_result1: {}".format(recall_result))
        rank_result = self.rank(query, recall_result) ## 返回降序排序的列表
        # rank_result = list(filter(lambda x:x[4] > 0.8, rank_result))

        logger.info("response: {}".format(rank_result))
        return rank_result
VEC_MODEL_PATH = "/home/workspace/NLP/jinguowei2/simcse-chinese-roberta-wwm-ext"
VEC_INDEX_DATA = "vec_index_test2023121301_20w"
class shuiwuRetrieveChain(Tool):
    # description = '个人办理内部退养手续而取得的一次性补贴收入如何计算缴纳个人所得税'
    description = '关于个人所得税的相关问题，税务征收问题以及和税务政策等相关的税务知识问题调用该工具'
    name = 'Document1'
    parameters: list = [{
        'name': 'question',
        'description': '用户的问题',
        'required': True
    }]
    def _local_call(self, *args, **kwargs):
        question = kwargs['question']
        searcher = Searcher(VEC_MODEL_PATH, VEC_INDEX_DATA,'index1')
        extra_knowledge = searcher.search(question)   #找到的所有相关的 知识 
        user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', args[0]).replace('{extra_knowledge}',extra_knowledge[0][1][1])
        #只用 相似度最大的做prompt拼接 
        # result = llm.original_generate(user_input)
        result = run_client("http://127.0.0.1:9090/llm_model",user_input)
        # return {"result": result,"immediate_result": extra_knowledge}  ##只要一个 返回给前端界面 
        # return {'result': f'模型的回答为:{result}，查询到的知识库内容为:{extra_knowledge}'}
        # return f'模型的回答为:{result}，查询到的知识库内容为:{extra_knowledge}'
        return result

class AMAPWeather(Tool):
    description = '获取对应城市的天气数据'
    name = 'amap_weather'
    parameters: list = [{
        'name': 'location',
        'description': 'get temperature for a specific location',
        'required': True
    }]

    def __init__(self, cfg={}):
        # self.cfg = cfg.get(self.name, {})

        # remote call
        self.url = 'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}'
        # self.token = self.cfg.get('token', os.environ.get('AMAP_TOKEN', ''))
        self.token = '451f1780d0d5ac1ac14ea48010d04f4d'
        self.city_df = pd.read_excel(
            'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/AMap_adcode_citycode.xlsx'
        )
        assert self.token != '', 'weather api token must be acquired through ' \
            'https://lbs.amap.com/api/webservice/guide/create-project/get-key and set by AMAP_TOKEN'

        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)

    def get_city_adcode(self, city_name):
        filtered_df = self.city_df[self.city_df['中文名'] == city_name]
        if len(filtered_df['adcode'].values) == 0:
            raise ValueError(
                f'location {city_name} not found, availables are {self.city_df["中文名"]}'
            )
        else:
            return filtered_df['adcode'].values[0]

    def __call__(self, *args, **kwargs):
        location = kwargs['location']
        response = requests.get(
            self.url.format(
                city=self.get_city_adcode(location), key=self.token))
        data = response.json()
        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            weather = data['lives'][0]['weather']
            temperature = data['lives'][0]['temperature']
            input = f'{location}的天气是{weather}温度是{temperature}度。'
            user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', args[0]).replace('{extra_knowledge}',input)
            result = run_client("http://127.0.0.1:9090/llm_model",user_input)
            # result = llm.original_generate(user_input)
            # return {'result': result}
            return result
# DEFAULT_TOOL_LIST = {"Document": DocumentRetrieveChain(),
#                      "SqlGenerator": SqlGeneratorChain(),
#                         "DatabaseQuery": DatabaseQueryChain(),
#                         "RenewInstance":AliyunRenewInstanceTool(),
#                         "Document1":shuiwuRetrieveChain(),
#                         "amap_weather":AMAPWeather(),
#                         "amap_poi_query":AMAPPOIQuery(),
#                         }
DEFAULT_TOOL_LIST = {"RenewInstance":AliyunRenewInstanceTool(),
                        "Document1":shuiwuRetrieveChain(),
                        "amap_weather":AMAPWeather(),
                        "amap_poi_query":AMAPPOIQuery(),
                        }
data = {'amap_weather':'{"name":"amap_weather","description":"获取对应城市的天气数据"}',
        'RenewInstance':'{"name":"RenewInstance","description":"续费一台包年包月ECS实例"}',
        'amap_poi_query':'{"name":"amap_poi_query","description":"获取对应地区地点的poi信息,你要把把我的地区地点名称连在一起输出作为一个参数,不能让他们之间用逗号分开。比如:问题是帮我查看杭州市的海底捞poi,你需要解析出来的参数是<杭州市海底捞>的参数，是连在一起的"}',
        'Document1':'{"name":"Document1","description":"关于个人所得税的相关问题，税务征收问题以及和税务政策等相关的税务知识问题调用该工具"}'}
# data = {'amap_weather':'{"name":"amap_weather","description":"获取对应城市的天气数据"}',
#         'RenewInstance':'{"name":"RenewInstance","description":"续费一台包年包月ECS实例"}',
#         'DatabaseQuery':'{"name":"DatabaseQuery","description":"数据库查询"}',
#         'Document':'{"name":"Document","description":"各个公司的招股书文档信息检索"}',
#         'amap_poi_query':'{"name":"amap_poi_query","description":"获取对应地区地点的poi信息,你要把把我的地区地点名称连在一起输出作为一个参数,不能让他们之间用逗号分开。比如:问题是帮我查看杭州市的海底捞poi,你需要解析出来的参数是<杭州市海底捞>的参数，是连在一起的"}',
#         'Document1':'{"name":"Document1","description":"关于个人所得税的相关问题，税务征收问题以及和税务政策等相关的税务知识问题调用该工具"}',
#         'SqlGenerator':'{"name":"SqlGenerator","description":"sql语句生成"}'}
def save_history_to_file(history, file_path='history.json'):
    with open(file_path, 'w') as f:
        json.dump(history, f)

def load_history_from_file(file_path='history.json'):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # 如果文件不存在，返回一个空列表
def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)
class BSAgentExecutor:

    def __init__(self,
                #  llm: LLM,
                 config,
                 display: bool = False):
        """
        the core class of ms agent. It is responsible for the interaction between user, llm and tools,
        and return the execution result to user.

        Args:
            llm (LLM): llm model, can be load from local or a remote server.
            tool_cfg (Optional[Dict]): cfg of default tools
            additional_tool_list (Optional[Dict], optional): user-defined additional tool list. Defaults to {}.
            prompt_generator (Optional[PromptGenerator], optional): this module is responsible for generating prompt
            according to interaction result. Defaults to use MSPromptGenerator.
            output_parser (Optional[OutputParser], optional): this module is responsible for parsing output of llm
            to executable actions. Defaults to use MsOutputParser.
            tool_retrieval (Optional[Union[bool, ToolRetrieval]], optional): Retrieve related tools by input task,
            since most of tools may be uselees for LLM in specific task.
            If is bool type and it is True, will use default tool_retrieval. Defaults to True.
            knowledge_retrieval (Optional[KnowledgeRetrieval], optional): If user want to use extra knowledge,
            this component can be used to retrieve related knowledge. Defaults to None.
        """

        self.llm = llm  #llm=Quen()
        self.config = config["config"]
        logger.info(self.config)
        self._init_tools()
        self.prompt_generator = BSPromptGenerator()
        self.output_parser = BSOutputParser()
        self.task_list = []
        self.task_no = None
        self.display = display
        self.agent_state = {}
        self.error_nums = 0

    def _init_tools(self):
        """init tool list of agent. We provide a default tool list, which is initialized by a cfg file.
        user can also provide user-defined tools by additional_tool_list.
        The key of additional_tool_list is tool name, and the value is corresponding object.

        Args:
            tool_cfg (Dict): default tool cfg.
            additional_tool_list (Dict, optional): user-defined tools. Defaults to {}.
        """
        self.available_tool_list = deepcopy(DEFAULT_TOOL_LIST)
        
#DEFAULT_TOOL_LIST = {"DocumentRetrieve": DocumentRetrieveChain(),
                   #SqlGeneratorChain(),
                       # "DatabaseQuery": DatabaseQueryChain()}

    def run(self,
            user_input: str,
            print_info: bool = False) -> List[Dict]:
        """ use llm and tools to execute task given by user

        Args:
            task (str): concrete task
            remote (bool, optional): whether to execute tool in remote mode. Defaults to False.
            print_info (bool, optional): whether to print prompt info. Defaults to False.

        Returns:
            List[Dict]: execute result. One task may need to interact with llm multiple times,
            so a list of dict is returned. Each dict contains the result of one interaction.
        """

        # no task, first generate task list
        
        
        idx = 0
        final_res = []
        addtional_tool={}
        cx = None
        self.reset()
        #总体来说，这段代码是一个重置方法，用于清除对象的状态。通过将某些属性设为默认值或空值，可以重置对象以准备执行新的操作或任务。
        self.agent_state["用户问题"] = user_input #self.agent_state = {} 在字典里里加入键值对
        # self.prompt_generator.init_plan_prompt(user_input)
        self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
        vec_model = VectorizeModel('/home/workspace/NLP/jinguowei2/simcse-chinese-roberta-wwm-ext')
        max_score = float('-inf')  # 初始最高分数为负无穷
        max_key = None
        for key, value in data.items():
            description = json.loads(value)['description']
            score = vec_model.predict_sim(user_input, description)
            print(description, score)  # 打印描述和对应的分数
            # logger.info("[DialogueManager] retrieval_result: {}".format(description))
            # logger.info(score)
            if score > max_score:
                max_score = score
                max_key = key
        print("最高分数的key:", max_key)
        # logger.info("最高分数的key:", max_key)
        logger.info("最高分数的key: {}".format(max_key))
        print("最高分数:", max_score)
        # logger.info("最高分数:", max_score)
        logger.info("最高分数: {}".format(max_score))
        if max_score>0.56:
            addtional_tool[max_key]=DEFAULT_TOOL_LIST[max_key]
            self.prompt_generator.init_task_prompt111(user_input, addtional_tool.values())  ##初始化 self.system_prompt1
            prompt111 = self.prompt_generator.generate1(self.task_no)
            cx =True
        else :
            result = run_client(self.config["llm_url"],user_input)
            return result

        #self.task_result_prompt = None
        #self.system_prompt = self.plan_template     系统提示 
        #self.user_prompt = self.user_template.replace("{user_question}",user_question)  用户问的问题 
        #self.prompt_generator = BSPromptGenerator() 

        while True:
            idx += 1
            # generate prompt and call llm  生成 prompt 然后 响应大模型
            llm_result, exec_result = '', {}
            prompt = self.prompt_generator.generate(self.task_no)   #return self.sep.join(prompt_list)
            #这个字符串是将系统提示和用户提示连接在一起，并在它们之间插入两个空行。 生成了一个完整的 prompt
#1你是一名高级督能助手，你可以先对间题进行分类，问题类型只有公司招跺书咨询和股票基金匏拒播查询两类，
#然后根据所给的信息列出回各该问题的任务列表。股票基金数据查询提供的表如下。 A最公司行业划分表，A股票日行情表,基金份额持有人结构，基金情券持仓明纽,基金可转债持仓明细，基金基本信息，基金日行情表，基金股票持仓明纽，基金规模变动表，港段票日行情表。faln用户问题、请帮我计算，在210165，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是﹖涨跂福幅是多少﹖百分兹保留两位小数。股票窃跌幅定义为:（收盘价-前一日收盘价/前一日收盘价）*100%。'

            #self.task_no = None
            #self.prompt_generator = BSPromptGenerator() 
            if cx:
                llm_result = self.llm.generate(prompt111)  ##自己生成 大模型的回复  根据提示 生成自己 的 generate方法 
                # llm_result = run_client(self.config["llm_url"],prompt111)
                if print_info:
                    print(f'|prompt{idx}: {prompt111}')
                    logger.info(f'|prompt{idx}: {prompt111}')
                    print(f'|llm_result{idx}: {llm_result}')
                    logger.info(f'|llm_result{idx}: {llm_result}')
            else:
                llm_result = self.llm.generate(prompt)
                # llm_result = run_client(self.config["llm_url"],prompt)
                if print_info:
                    print(f'|prompt{idx}: {prompt}')
                    logger.info(f'|prompt{idx}: {prompt}')
                    print(f'|llm_result{idx}: {llm_result}')
                    logger.info(f'|llm_result{idx}: {llm_result}')
            # if print_info:
            #     print(f'|prompt{idx}: {prompt}')
            #     print(f'|llm_result{idx}: {llm_result}')
            # parse and get tool name and arguments
            #self.output_parser = BSOutputParser()
            action, action_args = self.output_parser.parse_response(llm_result)
            #return action, parameters：这行代码返回工具名称和参数的元组。  
            if print_info:
                print(f'|action: {action}, action_args: {action_args}')
                logger.info(f'|action: {action}, action_args: {action_args}')
            final_res.append(llm_result)
            if action is None:
                #带有历史的回复 后续开一个 带历史的 大模型端口
                # global_history = load_history_from_file('/home/workspace/NLP/jinguowei2/app/chains/history.json')
                # result,global_history = llm.sql_generate(user_input,global_history)
                # save_history_to_file(global_history, '/home/workspace/NLP/jinguowei2/app/chains/history.json')
                # return {"result": result,"history":global_history}
                # return {'result': f'模型的回答为:{result}，用户对话历史为:{global_history}'}
                result = run_client(self.config["llm_url"],user_input)
                return result
            elif action in self.available_tool_list and addtional_tool:
                if not hasattr(action_args, 'items'):
                    if self.error_nums < 3:
                        self.error_nums += 1
                        #self.prompt_generator.init_plan_prompt(user_input)  #继续 初始化 从头再来一遍直到没错 如果大于三次还是错误的 那就直接 返回了
                            #最后的 问题对应的answer也是抛出的异常 
                        self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                        self.task_no = None
                        continue
                    return final_res
                action_args = self.parse_action_args(action_args)
                #parse_action_args 函数的作用是将工具执行所需的参数中的特殊标记替换为相应的包装器对象，
                #以便工具能够正确处理这些参数。它返回一个字典，其中包含处理后的参数名称和对应的参数值。
                tool = self.available_tool_list[action] #action 是下面字典里的一个键 
               # DEFAULT_TOOL_LIST = {"DocumentRetrieve": DocumentRetrieveChain(),
                    # "SqlGenerator": #SqlGeneratorChain(),
                   #     "DatabaseQuery":# DatabaseQueryChain()}
                try:
                    exec_result = tool(user_input,**action_args) # 传入 参数 得到的是 ；两个回答 一个是 大模型的回复 一个查询到的结果 
                    if print_info:
                        print(f'|exec_result: {exec_result}')
                        logger.info(f'|exec_result: {exec_result}')
                    # parse exec result and store result to agent state
                    if exec_result.get("error") is not None:  #有错误 才会进入  没错误不会进入 给出大模型 三次试错的机会
                        final_res.append(exec_result)
                        if self.error_nums < 3:
                            self.error_nums += 1
                            #self.prompt_generator.init_plan_prompt(user_input)  #继续 初始化 从头再来一遍直到没错 如果大于三次还是错误的 那就直接 返回了
                            #最后的 问题对应的answer也是抛出的异常 
                            self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                            self.task_no = None
                            continue
                        return final_res
                    # final_res.append(exec_result.get('result', ''))
                    else:
                        return exec_result
                
                    # self.parse_exec_result(exec_result)
                    #在这个示例中，代理状态 self.agent_state 中新增了一个键值对，键为 "任务1的返回结果"，值为 "执行成功"。
                    #这表示任务编号为 1 的任务的执行结果是 "执行成功"。通过将执行结果存储在代理状态中，可以在后续的代码中方便地访问和使用这个执行结果。
                except Exception as e:
                    exec_result = f'Action call error: {action}: {action_args}. \n Error message: {e}'
                    final_res.append({'error': exec_result})
                    if self.error_nums < 3:
                        self.error_nums += 1
                        #self.prompt_generator.init_plan_prompt(user_input)
                               # self.current_task_prompt = None   重置为None
                    #self.task_result_prompt = None  重置为None
                        self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                        self.task_no = None
                        continue
                    return final_res
            else:  ##action不在 给定的 已知的 三个 工具里面  #return action, parameters：这行代码返回工具名称和参数的元组。  
                exec_result = f"Unknown action: '{action}'. "
                final_res.append({'error': exec_result})
                answer = run_client(self.config["llm_url"],user_input)
                # answer = self.llm.generate(user_input)
                # if self.error_nums < 3:
                #     self.error_nums += 1
                #     # self.prompt_generator.init_plan_prompt(user_input)
                #     self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                #     self.task_no = None
                #     continue
                return answer

            # display result
            if self.display:
                display(llm_result, exec_result, idx)
            
            # return llm_result
           # 总结起来，这段代码定义了一个用于在 Jupyter Notebook 中显示每一轮执行结果的函数。
            #它根据结果的类型和格式，使用适当的方式进行显示，包括显示 JSON 数据和漂亮的格式化文本。这样可以在 Notebook 中更好地展示和查看执行结果。
            # if self.task_no is None:
            #     #self.prompt_generator = BSPromptGenerator()
            #     self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
            #     #self.task_result_prompt = None  值变成None
            #     #更新 self.system_prompt 的值为自行调用工具 然后 如果符合加前后标志符
            #     self.task_no = 0
            #     self.task_list = [task.strip() for task in re.split(r'\n[0-9].',llm_result)[1:]]
            # #总结起来，这段代码的作用是从给定的 llm_result 字符串中提取任务列表。它通过正则表达式拆分字符串，并对拆分结果进行处理，去除首尾空白字符，
            #     #然后将处理后的任务列表赋值给 self.task_list 属性。这样可以方便地获取和使用从 llm_result 中提取的任务列表。
            # #如果不是 None才会执行下面的 else
            # else:
            #     self.task_no += 1
            #     if self.task_no >= len(self.task_list):    #self.task_list 中存入的是 大模型的 第一个回答 拆分后的关键问题的列表 
            #         return final_res
            # #self.prompt_generator = BSPromptGenerator()
            # try:
            #     self.prompt_generator.update_task_prompt(self.task_list[self.task_no])
            # except Exception as e:
            #     return final_res

            #result列表中的 第一个问题拿出来了 self.task_list[0] 传入update_task_prompt函数组装成了一个 行的 prompt 
            #“当前任务：查询中信行业分类划分的一级行业列表”
            #self.current_task_prompt  将self.current_task_prompt 变成一个 prompt


    def reset(self):
        """
        clear history and agent state
        """
        # self.prompt_generator.reset()
        self.task_no = None   #将对象的 task_no 属性设置为 None。这个属性可能表示当前任务或操作的编号。通过将其设为 None，可以清除任务编号并重置对象的状态。
        self.exec_result = ''   #将对象的 exec_result 属性设置为空字符串。这个属性可能表示执行的结果或操作的输出。通过将其设为空字符串，可以清除执行结果并重置对象的状态。
        self.agent_state = dict()  #将对象的 agent_state 属性设置为空字典。这个属性可能表示代理或对象的状态信息。通过将其设为空字典，可以清除代理状态并重置对象的状态。
        self.error_nums = 0   #将对象的 error_nums 属性设置为 0。这个属性可能表示错误的数量或计数。通过将其设为 0，可以重置错误计数并清除对象的状态。

    def parse_action_args(self, action_args):
        """
        replace action_args in str to Image/Video/Audio Wrapper, so that tool can handle them
        """
        parsed_action_args = {}
        for name, arg in action_args.items():
            try:
                true_arg = self.agent_state.get(arg, arg)
            except Exception:
                true_arg = arg
            parsed_action_args[name] = true_arg
        return parsed_action_args

    def parse_exec_result(self, exec_result):
        """
        update exec result to agent state.
        key is the str representation of the result.
        """
        self.agent_state[f"任务{self.task_no+1}的返回结果"] = exec_result["result"]
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
        answer = self.dialogue_manager.run(json_decode(self.request.body).get("query", ""),print_info=True)
        logger.info(answer)
        # response_body = {"answer": answer}
        response_body = answer
        self.write(response_body)

def StartDialogueManagerHandler(request_config: dict, dialogue_manager: BSAgentExecutor):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], DialogueManagerHandler, {"dialogue_manager":dialogue_manager})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"],address='192.168.0.110')
    tornado.ioloop.IOLoop.current().start()
class SearcherHandler(RequestHandler):
    
    def initialize(self, searcher:Searcher):
        self.searcher = searcher
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":[{
        #         "match_query":"match_query",
        #         "answer":"answer",
        #         "score":"score"
        #     }]
        # }

    async def post(self):
        answers = self.searcher.search(json_decode(self.request.body).get("query", ""))
        result = []
        for answer in answers:
            tmp_result = {}
            # tmp_result["query"] = answer[0]
            tmp_result["answer"] = answer[1][1]["answer"]
            tmp_result["match_query"] = answer[1][0]
            tmp_result["score"] = str(answer[3])
            result.append(copy.deepcopy(tmp_result))
        response_body = {"answer": result}
        self.write(response_body)

def StartSearcherHandler(request_config: dict, searcher: Searcher):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], SearcherHandler, {"searcher":searcher})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()
def launch_service(config, model_mode):
    # if model_mode == "llm_model":
    #     # 解决windows下多进程使用pt会导致pickle序列化失败的问题，https://blog.csdn.net/qq_21774161/article/details/127145749
    #     llm_model = Agentmodel(config["process_llm_model"]["model_path"])
    #     # agent_model = BSAgentExecutor(llm_model)
    #     StartLlmHandler(config["process_llm_model"], llm_model)
    #     # processes = [process_llm_model]
        # for process in processes:
        #     process.start()
        # for process in processes:
        #     process.join()
    # elif model_mode == "searcher":
    #     # 这一行创建了一个Process对象，用于启动一个新进程来运行搜索器处理服务。Process是Python多进程模块multiprocessing的一部分，它允许程序并行地运行多个任务。
    #     # target=StartSearcherHandler: 指定进程启动时调用的函数。在这里，StartSearcherHandler函数可能是设置和运行HTTP服务的函数，类似于在之前的LlmHandler示例中的StartLlmHandler
    #     # args=(config["process_searcher"], searcher): StartSearcherHandler函数的参数。第一个参数是包含进程搜索配置的字典，第二个参数是刚才创建的searcher实例
    #     searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"])
    #     StartSearcherHandler(config["process_searcher"], searcher)
        # process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))
        
    if model_mode == "dialogue":
        # 这一行创建了一个Process对象，用于启动一个新进程来运行搜索器处理服务。Process是Python多进程模块multiprocessing的一部分，它允许程序并行地运行多个任务。
        # target=StartSearcherHandler: 指定进程启动时调用的函数。在这里，StartSearcherHandler函数可能是设置和运行HTTP服务的函数，类似于在之前的LlmHandler示例中的StartLlmHandler
        # args=(config["process_searcher"], searcher): StartSearcherHandler函数的参数。第一个参数是包含进程搜索配置的字典，第二个参数是刚才创建的searcher实例
        # searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"])
        # process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))

        dialogue_manager = BSAgentExecutor(config["process_dialogue_manager"])
        StartDialogueManagerHandler(config["process_dialogue_manager"], dialogue_manager)
        # process_dialogue_manager = Process(target=StartDialogueManagerHandler, args=(config["process_dialogue_manager"], dialogue_manager))
        # vec_model = VectorizeModel(config["process_vec_model"]["VEC_MODEL_PATH"])
        # process_vec_model = Process(target=StartVecModelHandler, args=(config["process_vec_model"], vec_model))

        # processes = [process_searcher]
        # processes = [process_searcher, process_dialogue_manager]
        # for process in processes:
        #     process.start()
        # for process in processes:
        #     process.join()
    # elif model_mode == "searcher":
    #     # 这一行创建了一个Process对象，用于启动一个新进程来运行搜索器处理服务。Process是Python多进程模块multiprocessing的一部分，它允许程序并行地运行多个任务。
    #     # target=StartSearcherHandler: 指定进程启动时调用的函数。在这里，StartSearcherHandler函数可能是设置和运行HTTP服务的函数，类似于在之前的LlmHandler示例中的StartLlmHandler
    #     # args=(config["process_searcher"], searcher): StartSearcherHandler函数的参数。第一个参数是包含进程搜索配置的字典，第二个参数是刚才创建的searcher实例
    #     searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"],'index1')
    #     StartSearcherHandler(config["process_searcher"], searcher)
    #     # process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))
    else:
        logger.info("init service error")
# def launch_service(config, model_mode):
#     if model_mode == "llm_model":
#         # 解决windows下多进程使用pt会导致pickle序列化失败的问题，https://blog.csdn.net/qq_21774161/article/details/127145749
#         llm_model = LlmModel(config["process_llm_model"]["model_path"], config["process_llm_model"]["model_config"])
#         StartLlmHandler(config["process_llm_model"], llm_model)
#         # processes = [process_llm_model]
#         # for process in processes:
#         #     process.start()
#         # for process in processes:
#         #     process.join()
#     elif model_mode == "searcher":
#         # 这一行创建了一个Process对象，用于启动一个新进程来运行搜索器处理服务。Process是Python多进程模块multiprocessing的一部分，它允许程序并行地运行多个任务。
#         # target=StartSearcherHandler: 指定进程启动时调用的函数。在这里，StartSearcherHandler函数可能是设置和运行HTTP服务的函数，类似于在之前的LlmHandler示例中的StartLlmHandler
#         # args=(config["process_searcher"], searcher): StartSearcherHandler函数的参数。第一个参数是包含进程搜索配置的字典，第二个参数是刚才创建的searcher实例
#         searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"])
#         StartSearcherHandler(config["process_searcher"], searcher)
#         # process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))
        
#     elif model_mode == "dialogue":
#         # 这一行创建了一个Process对象，用于启动一个新进程来运行搜索器处理服务。Process是Python多进程模块multiprocessing的一部分，它允许程序并行地运行多个任务。
#         # target=StartSearcherHandler: 指定进程启动时调用的函数。在这里，StartSearcherHandler函数可能是设置和运行HTTP服务的函数，类似于在之前的LlmHandler示例中的StartLlmHandler
#         # args=(config["process_searcher"], searcher): StartSearcherHandler函数的参数。第一个参数是包含进程搜索配置的字典，第二个参数是刚才创建的searcher实例
#         # searcher = Searcher(config["process_searcher"]["VEC_MODEL_PATH"], config["process_searcher"]["VEC_INDEX_DATA"])
#         # process_searcher = Process(target=StartSearcherHandler, args=(config["process_searcher"], searcher))

#         dialogue_manager = DialogueManager(config["process_dialogue_manager"])
#         StartDialogueManagerHandler(config["process_dialogue_manager"], dialogue_manager)
#         # process_dialogue_manager = Process(target=StartDialogueManagerHandler, args=(config["process_dialogue_manager"], dialogue_manager))
#         # vec_model = VectorizeModel(config["process_vec_model"]["VEC_MODEL_PATH"])
#         # process_vec_model = Process(target=StartVecModelHandler, args=(config["process_vec_model"], vec_model))

#         # processes = [process_searcher]
#         # processes = [process_searcher, process_dialogue_manager]
#         # for process in processes:
#         #     process.start()
#         # for process in processes:
#         #     process.join()
#     else:
#         logger.info("init service error")


if __name__ == "__main__":
    config = {
            #  "process_llm_model":{"port":9094, 
            #                           "url_suffix":"/llm_model", 
            #                         #   "model_path":"/home/extra1T/kanghong/chatglm3-6b-32k",
            #                         #   "model_path":"/home/extra1T/load_model/chatglm3-6b",
            #                           "model_path":"/home/extra1T/load_model/ModelScope-Agent-7B",
            #                           "model_config":{}},
             "process_dialogue_manager":{"port":9093,
                                      "url_suffix":"/dialogue_manager",
                                      "config":{"search_url":"http://127.0.0.1:9090/searcher",
                                                "llm_url":"http://127.0.0.1:9090/llm_model"}}
    }
    launch_service(config, sys.argv[1])
# if __name__ == "__main__":
#     config = {"process_searcher":{"port":9090, 
#                                       "url_suffix":"/searcher", 
#                                       "VEC_MODEL_PATH":"/home/workspace/NLP/jinguowei2/simcse-chinese-roberta-wwm-ext",
#                                       "VEC_INDEX_DATA":"vec_index_test2023121301_20w"},
#              "process_vec_model":{"port":9091, 
#                                       "url_suffix":"/vec_model", 
#                                       "VEC_MODEL_PATH":"/home/extra1T/load_model/simcse-chinese-roberta-wwm-ext"},
#              "process_llm_model":{"port":9094, 
#                                       "url_suffix":"/llm_model", 
#                                     #   "model_path":"/home/extra1T/kanghong/chatglm3-6b-32k",
#                                       "model_path":"/home/extra1T/load_model/chatglm3-6b",
#                                       "model_config":{}},
#              "process_dialogue_manager":{"port":9093,
#                                       "url_suffix":"/dialogue_manager",
#                                       "config":{"search_url":"http://127.0.0.1:9090/searcher",
#                                                 "llm_url":"http://127.0.0.1:9092/llm_model"}}
#     }
#     launch_service(config, sys.argv[1])