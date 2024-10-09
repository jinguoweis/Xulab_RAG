from tools import Tool
from apis.dataset_api import sqldb
# from apis.model_api import llm
from prompt.bs_prompt import BS_CHAT_SQLRESULT_TEMPLATE
import re
import json
import requests
# from utils.post_process_sql_result import post_process_answer
p = "\d+\.\d+"
bd = '[=,.?!@#$%^&*()_+:"<>/\[\]\\`~——，。、《》？；’：“【】、{}|·！￥…（）-]'
weishu_dict = {
    "1":1,
    "2":2,
    "3":3,
    "4":4,
    "5":5,
    "一":1,
    "二":2,
    "三":3,
    "四":4,
    "五":5,
    "两":2
}
def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)
def refine_xiaoshu(answer, ji):
    a = answer.strip()
    a_li = a.split(".")
    xiaoshu_part = re.match("\d+", a_li[-1]).group()
    new_xiaoshu_part = ""
    cur = 0
    while cur < weishu_dict[ji]:
        if len(xiaoshu_part) > cur:
            new_xiaoshu_part += xiaoshu_part[cur]
        else:
            new_xiaoshu_part += "0"
        cur += 1
    a_li[-1] = a_li[-1].replace(xiaoshu_part, new_xiaoshu_part)
    return ".".join(a_li)

def post_process_answer(qustion, answer):
    ans = answer.replace("\"", "")
    ans = ans.replace("[", "").replace("]", "").replace("(","").replace(")","")
    q = qustion
    if "费率" in q and len(ans.split(",")) == 1 and '%' not in ans:
        ans += '%'
    if "百分比" in q and len(ans.split(",")) == 1 and '%' not in ans:
        ans += '%'
    if "涨跌幅" in q and '%' not in ans:
        m = re.search(p, ans)
        if m:
            ans = ans.replace(m.group(), m.group()+'%')
    if "取整" in q and ".0" in ans:
        ans = ans.replace('.0', "")
    q_li = re.split(bd, q)
    for sub_q in q_li:
        if "小数" in sub_q:
            if "不超过" in sub_q:
                continue
            else:
                for ji in weishu_dict:
                    if ji+"位" in sub_q:
                        ans_list = ans.split(",")
                        if len(ans_list) == 1:
                            ans_list[-1] = refine_xiaoshu(ans,ji)
                        else:
                            for i,a0 in enumerate(ans_list):
                                if a0.find(".") > 0:
                                    ans_list[i] = refine_xiaoshu(a0,ji)
                                    break
                        ans = ", ".join(ans_list)
                        break
    return ans
class DatabaseQueryChain(Tool):
    description = '数据库查询'
    name = 'DatabaseQuery'
    parameters: list = [{
        'name': 'sql_sentence',
        'description': 'sql语句',
        'required': True
    }]

    def _local_call(self, *args, **kwargs):
        if kwargs.get("sql_sentence") is None:
            return {"error": "参数错误，请检查!sql_sentence: None"}
        sql_result = sqldb.select_data(kwargs['sql_sentence'])
        # sql_result = " ".join([" ".join([str(rr) for rr in res]) for res in sql_result])
        sql_result = post_process_answer(args[0], str(sql_result))
        user_input = BS_CHAT_SQLRESULT_TEMPLATE.replace('{user_question}', args[0]).replace('{sql_result}',sql_result)
        # result = llm.original_generate(user_input)
        result = run_client("http://127.0.0.1:9092/llm_model",user_input)
        return {"result": result, "immediate_result": sql_result}
