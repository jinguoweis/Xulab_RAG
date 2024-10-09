# coding=utf-8
# Filename:    llm_model.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型调用模块，这里默认用的chatglm2

from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain   
prompt_template = """你是一个负责将用户问题进行分类的智能决策机器人。你需要根据问题的内容推导出它所属的类别。类别仅限于以下三个：天气，税务知识，闲聊。请根据用户的问题提供思路，并最终返回准确的分类。

分类类别如下：
1. 天气
2. 税务知识
3. 闲聊

你需要一步一步推理问题的类别，并且你只能从上面的三类中选择其中一个进行回答。请参考我给出的示例进行回答，只回答出类别，不要用超一个词进行回答：
<example>

### 示例 1
问题：今天的天气怎么样？
思路：问题涉及天气状况，因此属于“天气”类别。
分类：天气

### 示例 2
问题：我需要了解如何申报个人所得税。
思路：问题明确涉及税务申报的相关内容，因此属于“税务知识”类别。
分类：税务知识

### 示例 3
问题：你喜欢什么类型的电影？
思路：这个问题与日常生活的随意对话相关，属于“闲聊”类别。
分类：闲聊

<example>
问题：{question}
类别回答："""

class QwenLlmModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",offload_folder="offload",trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
        # from utils import load_model_on_gpus
        # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        self.model = self.model.eval()
    def predict(self, query):
        rag_chain = LLMChain(
        llm=self.llm,
        prompt=self.prompt)
        raw_answer = rag_chain.run({ "question": query})
        return raw_answer
if __name__ == "__main__":
    path = "/home/extra1T/model_embeding/qwen/Qwen2-7B"
    llm_model = QwenLlmModel(path)
    print(llm_model.predict("今天吃什么？"))