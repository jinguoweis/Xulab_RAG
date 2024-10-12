# Hands on XuLab_RAG
## 项目介绍🤗🤗🤗
RAG (Retrieval-Augmented Generation) 是一种结合检索与生成模型的技术，常用于问答系统、对话生成和知识增强等任务。它的主要思想是通过信息检索从知识库中获取相关文档，并将这些检索到的信息作为上下文，辅助生成模型 (通常是大型语言模型) 生成更加准确、相关的答案。本项目旨在开源一个完整的RAG系统，以帮助大家从底层上了解整个RAG构建的具体流程。在此基础上，我们将Agent也集成于我们所构建的RAG系统中，以增加用户对于外部工具调用的体验。详细的优化策略以及集成方法在以下小节会逐一介绍。
# 项目整体架构图
![image](https://github.com/jinguoweis/Xulab_RAG/blob/master/RAG.png)
# 项目结构
```plaintext
├── agent
│   ├── bs_agent.py
│   ├── output_parser.py
│   ├── output_wrapper.py
├── apis
│   ├── dataset_api.py
│   ├── model_api.py
│   └── retrieve_api.py
├── assistance
│   ├── console
│   │   ├── dialogue_manager.py
│   │   ├── dialogue_manager_copy.py
│   │   ├── output_parser.py
│   │   └── output_wrapper.py
│   ├── handles
│   │   ├── Qwen2_llm_handler.py
│   │   ├── dialogue_manager_handler.py
│   │   ├── llm_handler.py
│   │   ├── modelscope_Agent_handler.py
│   │   └── search_handler.py
│   ├── models
│   │   ├── Qwen2_llm_model.py
│   │   ├── chatglm3_model.py
│   │   └── modelscope_Agent_model.py
│   ├── searcher
│   │   └── search.py
│   ├── client.py
│   ├── main_service_online.py
│   ├── run.sh
│   └── webui.py
├── chains
│   ├── __init__.py
│   ├── amap_weather_chain.py
│   ├── database_query_chain.py
│   ├── doc_retrieve_chain.py
│   └── sql_generator_chain.py
├── config
│   ├── __init__.py
│   ├── base_config.py
│   └── model_config.py
├── data
│   ├── pdf_txt_new
│   ├── answer.jsonl
│   ├── answers.csv
│   ├── corpus copy.txt
│   ├── final_contexts_question.json
│   ├── question.json
│   └── test_question.csv
├── logs
│   ├── dialogue_manager.log
│   ├── llm_model.log
│   ├── modelscope_Agent_model.log
│   ├── qwen2_llm_model.log
│   └── searcher.log
├── model
│   ├── llm
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── qwen.py
│   ├── vec_model
│   │   ├── simcse_model.py
│   │   ├── vec_model.py
│   └── llm_factory.py
├── prompt
│   ├── __init__.py
│   ├── bs_prompt.py
│   └── prompt.py
├── retrieve
│   ├── RAG
│   │   ├── data
│   │   │   ├── answer.csv
│   │   │   ├── answers.csv
│   │   │   ├── corpus copy.txt
│   │   │   ├── final_contexts.json
│   │   │   ├── final_contexts_question.json
│   │   │   ├── final_contexts_rewrite.json
│   │   │   ├── output_answers_query_rewrite.csv
│   │   │   ├── test_question.csv
│   │   │   ├── test_question_all.csv
│   │   │   ├── test_question_query_rewrite.csv
│   │   ├── save_bm25
│   │   │   ├── model.pkl
│   │   ├── save_faiss
│   │   │   ├── index.faiss
│   │   │   ├── index.pkl
│   │   ├── bge_reanker.py
│   │   ├── bm25&faiss.py
│   │   ├── bm25.py
│   │   ├── faiss1.py
│   │   ├── generate_context.py
│   │   ├── main.py
│   │   ├── model_generate.py
│   │   ├── model_generate_query_write.py
│   │   ├── query_rewrite_data_process.py
│   │   ├── rag_retrieve.py
│   │   ├── rrf_fusion_phj.py
│   │   └── utils_process.py
│   └── doc_retrieve.py
├── tools
│   ├── __init__.py
│   ├── database_query_tool copy.py
│   ├── database_query_tool.py
│   ├── doc_retrieve_tool.py
│   ├── llm_chat_tool.py
│   ├── sqlite.py
│   └── tool.py
├── utils
│   ├── __init__.py
│   ├── file_processor.py
│   ├── models_download.py
│   ├── pdf2txt.py
│   ├── post_process_sql_result.py
├── RAG.png
├── README.md
└── agent_predict.py
```
# QuickStrat
安装依赖，需要 Python 3.10 以上版本。
```plaintext
pip install -r requirements.txt
```
# RAG构建和优化策略
对于RAG的构建及优化策略在retrieve文件夹中有详细的代码展示。此外，为了方便大家能够更好的理解构建流程，我们将我们所参与的2024年科大讯飞的RAG的智能问答挑战赛的部分问答数据和提取到的上下文上传到了retrieve/RAG/data目录下，供大家直接使用并且一键启动代码。😇😇😇
# 一键启动RAG
```plaintext
python main.py
```
这里在单独给大家详细的介绍一下各个py文件的具体含义：
- utils_process.py中展示的是各种工具函数的集合
- bm25.py展示的是我们使用bm25进行关键词匹配召回的逻辑
- faiss1.py展示的是我们使用faiss向量库进行语义检索召回的策略
- rrf_fusion_phj.py展示的是多路召回后的融合逻辑
- model_generate_query_write.py展示的是我们进行查询优化的策略(Query Rewrite、Hyde)
- model_generate.py展示的是我们使用基于各种融合策略后提取的上下文信息
- bge_reanker.py展示的是我们使用基于bge-reanker-large进行重排后的精排逻辑。
大家可以根据自身需求单独运行上述文件，了解原理😃😃😃
# Agent的集成策略
对于Agent的集成，我们的基座采用阿里开源的魔搭Agent进行工具参数的解析和定义。同样的为了方便大家更好的了解和感受到Agent带来的巨大魔力😈😈😈，我们将我们所参与的天池2023博金大模型挑战赛的的部分数据上传到了/data目录下，供大家直接使用并且一键启动Agent代码。😜😜😜
# 一键启动Agent
```plaintext
python agent_predict,py
```
- 那么在这里我该写什么了呢？相信一些聪明的小伙伴已经猜出来了。那当然是给大家再详细的介绍一下Agent的具体实现流程，那么话不多说，我们往下看。🤪有的小伙伴可能有些疑问，你这不就是把RAG和Agent分别调用了一下吗？答案当然不是的，在之前的基础上我们作了一些创新和改进。
- 我们将以往的外接的RAG知识库集成到了我们的Agent当中，具体来说我们把基于私有领域的数据查询，也就是外接的RAG封装成了一个具体的tools，在本项目中，我们的外接RAG的知识库用的是税务领域的私有数据，大家可以根据自己的数据进行封装索引和保存。chains/文件夹中存放了本项目所使用的三种工具，分别是天气，税务RAG查询以及闲聊工具。
- 详细的代码执行逻辑，大家可以研读assistance/console/dialogue_manager.py。这里再详细的说明一点，为了更加准确的根据用户的问题去响应具体的意图，我们采用的是基于Qwen2使用LLMRouterChain方法进行用户的意图识别分析和路由选择的。代码展示如下：
```plaintext
    def route_to_chain(self,route_name,query):
        if '闲聊' == route_name['answer']:
            result = run_client("http://127.0.0.1:9092/llm_model", query)
            # result = result['answer']
            return [0,result]
        elif "天气" == route_name['answer']:
            return ['amap_weather',DEFAULT_TOOL_LIST['amap_weather']]
        else:
            return ['Document1',DEFAULT_TOOL_LIST['Document1']]
```
对于prompt的构建，我们采用的是COT，Few-shot等提示词工程，详细的构建大家也可以在代码中进行研究。
# 服务分拆及模型模块
最后，我们重点聊一下服务分拆，这里我能想到两个方案，一种是把向量模型和索引拆开成两个服务，另一种是合成一个，这两个方案各有各的优势：
- 将模型和索引拆分部署带来了更好的可维护性，便于分别进行模型更新和索引管理。这样，模型不仅可以用于向量召回，还能在其他任务中灵活应用。同时，考虑到向量模型通常需要显卡支持，单独部署能够更合理地分配硬件资源。因此，从系统设计的角度来看，模型单独部署更为合适。
- 不过，如果项目相对简单，模型仅用于少量向量检索且对性能要求不高，或者没有显卡支持，将模型和索引放在同一个服务中也并非不可行。此外，由于向量模型和检索过程之间存在一定的耦合关系，模型更新往往伴随着索引更新，因此两者在同一服务中一起更新也不失为一种合理的选择。
- 最终，奔着简单的原则，我们合并成了一个服务。对于服务的构，我们采用的是的tornado框架，当然更推荐的还有fastapi之类的，也可以用，这里需要一些tornado基础，事后有兴趣大家可以专项学，不过这里要求的不高，大家看明白应该不难。这里的核心是，把每个服务写到handler里面，然后服务启动的时候分别把这些handler加载就好了。这部分代码主要展示在/assistance中，老样子，我还是给大家详细的介绍一下其中的代码结构：console/dialogue_manager.py展示的是最终集成RAG和Agent的中控模块，/handles中展示的是本项目用到的各个模型和搜索模块以及中控模块的服务启动的代码。/models中展示的是对于各个模型的初始化定义，/search中展示的是各种执行各种优化策略和检索的初始化定义。
# 注意点：
- 继承RequestHandler，内部其实只需要initialize和post就行，注意这里的post需要使用异步async。
- 必要的输入和输出，为了方便阅读，可以写在注释里。
- StartSearcherHandler主要是为了方便服务启动调用，当然写在外面统一写，其实也是可以的。
- 注意这里的所有类，尤其是模型之类的，一定要传进来，而不能在内部直接启动，因为tornado里的initialize，是每次请求都会执行一次的，这不是常规的python类下的__init__函数。
- 最后就是服务的启动，这个就是一个脚本即可完成，说白了就是各自加载，装载好后多进程启用。
# 一键开启服务
```plaintext
bash run.sh
```
既然有了服务端，为了请求服务，肯定也需要客户端，这个也比较简单，就是一个request就能模拟了。
```plaintext
import numpy as np
import json,requests,time,random
from multiprocessing import Pool

def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)

if __name__ == "__main__":
    url = "http://127.0.0.1:9090/searcher"
    print(run_client(url, "女朋友生气了怎么办？")) # 单元测试
```
此外，大家开服务后可以一键启动webui.py进行展示
# 网页版对话demo
```plaintext
python webui.py
```
# 最终启动demo结果如下:
![image](https://github.com/jinguoweis/Xulab_RAG/blob/master/RAG.png)
# 总结
本项目开源了一个基于税务私有领域数据的RAG+Agent的问答系统，旨在帮助用户快速获取税务信息并提升交互体验。系统采用前后端分离的架构，基于Tornado框架设计，以提高系统的维护性和可扩展性。
以下是我们所构思和实现的具体步骤：
- 模型微调：对税务数据使用Lora技术对ChatGLM-3进行SFT微调，增强其对税务查询的专业性和准确性。
- RAG架构设计：结合多路径检索技术，采用BM25与Faiss向量搜索，利用bge-rerank-large模型优化结果排序，确保高相关性返回。
- 查询优化：通过Query Rewrite与HyDE技术对用户查询进行预处理，提升检索效果。
- Prompt工程：使用COT、TOT、Few-shot、Step Back等提示词工程来设计prompt生成问题的任务列表和任务序号，利用大语言模型多次循环调用生成能力，以处理复杂和多样化的用户提问。
- RouterChain及Agent调用:基于Qwen2使用LLMRouterChain进行用户的意图识别分析和路由选择，基于魔搭Agent进行工具的调用和定义，提升用户体验与交互效率。
服务拆分与模块化：基于Tornado框架将索引查询与 LLM 响应拆分为两个子服务，增强系统的模块化和可操作性。
# 参考文献
[https://mp.weixin.qq.com/s/RONG0mK07ZHrQZ5mgr31cg](#aaa)
