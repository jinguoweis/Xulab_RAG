# Hands on XuLab_RAG
## 项目介绍🤗🤗🤗
RAG (Retrieval-Augmented Generation) 是一种结合检索与生成模型的技术，常用于问答系统、对话生成和知识增强等任务。它的主要思想是通过信息检索从知识库中获取相关文档，并将这些检索到的信息作为上下文，辅助生成模型 (通常是大型语言模型) 生成更加准确、相关的答案。本项目旨在开源一个完整的RAG系统，以帮助大家从底层上了解整个RAG构建的具体流程。在此基础上，我们将Agent也集成于我们所构建的RAG系统中，以增加用户对于外部工具调用的体验。详细的优化策略以及集成方法在以下小节会逐一介绍。
# 项目整体架构图
![image](https://github.com/jinguoweis/Xulab_RAG/blob/master/RAG.png)
# 项目结构
```plaintext
tinyRAG
├─ build.ipynb
├─ component
│  ├─ chain.py
│  ├─ databases.py
│  ├─ data_chunker.py
│  ├─ embedding.py
│  └─ llms.py
├─ data
│  ├─ dpcq.txt
│  ├─ README.md
│  └─ 中华人民共和国消费者权益保护法.pdf
├─ db
│  ├─ document.json
│  └─ vectors.json
├─ image
│  └─ 5386440326a2c9c5a06b5758484d375.png
├─ push.bat
├─ README.md
├─ requirements.txt
└─ webdemo_by_gradio.ipynb
```
# QuickStrat
安装依赖，需要 Python 3.10 以上版本。
```plaintext
pip install -r requirements.txt
```
# RAG构建和优化策略
对于RAG的构建及优化策略在retrieve文件夹中有详细的代码展示。此外，为了方便大家能够更好的理解构建流程，我们将2024年科大讯飞的RAG的智能问答挑战赛的部分问答数据和提取到的上下文上传到了retrieve/RAG/data目录下，供大家直接使用并且一键启动代码。😇😇😇
# 一键启动
```plaintext
python main.py
```
这里在单独给大家详细的介绍一下各个py文件的具体含义：utils_process.py中展示的是各种工具函数的集合，bm25.py展示的是我们使用bm25进行关键词匹配召回的逻辑，faiss1.py展示的是我们使用faiss向量库进行语义检索召回的策略，rrf_fusion_phj.py展示的是多路召回后的融合逻辑，model_generate_query_write.py展示的是我们进行查询优化的策略(Query Rewrite、Hyde),model_generate.py展示的是我们使用基于各种融合策略后提取的上下文信息，bge_reanker.py展示的是我们使用基于bge-reanker-large进行重排后的精排逻辑。大家可以根据自身需求单独运行上述文件，了解原理😃😃😃
# Agent的集成策略
