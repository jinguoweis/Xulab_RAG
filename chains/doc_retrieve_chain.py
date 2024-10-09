from tools import Tool
# from apis.retrieve_api import doc_retrieve
# from apis.retrieve_api import doc_retrieve

# from apis.model_api import llm
from prompt.bs_prompt import BS_CHAT_KNOWLEDGE_TEMPLATE
import requests,json

def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)


class DocumentRetrieveChain(Tool):
    description = '关于个人所得税的相关问题，税务征收问题以及和税务政策等相关的税务知识问题调用该工具'
    name = 'Document1'
    parameters: list = [{
        'name': 'company',
        'description': '用户的问题',
        'required': True
    }]


    def _local_call(self, *args, **kwargs):
        url1 = "http://127.0.0.1:9091/searcher"
        extra_knowledge = run_client(url1, args[0])['answer'][0]['context'] # 单元测试
        # extra_knowledge = doc_retrieve.get_single_extra_knowledge(kwargs.get("company",""),args[0])
        user_input = BS_CHAT_KNOWLEDGE_TEMPLATE.replace('{user_question}', args[0]).replace('{extra_knowledge}',extra_knowledge)
        url2 = "http://127.0.0.1:9092/llm_model"
        result = run_client(url2, user_input)
        # result = llm.original_generate(user_input)
        return {"result": result,"immediate_result": extra_knowledge}
        # return {"回答是:": result,"检索到的信息为:": extra_knowledge}