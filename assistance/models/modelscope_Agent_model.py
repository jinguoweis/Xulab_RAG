# coding=utf-8
# Filename:    llm_model.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型调用模块，这里默认用的chatglm2

from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
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