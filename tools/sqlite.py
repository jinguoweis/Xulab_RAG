import sqlite3
from typing import List, Tuple
from config import logger, DB_FILE_PATH
import sys
sys.path.append('/home/extra1T/jin/GoMate/RAG')
# from utils.post_process_sql_result import post_process_answer
import re
import json
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
class Sqldb:
    def __init__(self,p_db_path: str) -> None:
        self.conn = sqlite3.connect(p_db_path)

    def __del__(self):
        self.conn.close()
    
    def commit(self) -> None:
        self.conn.commit()

    # 创建表
    def create_table(self,p_tb_name: str,p_tb_columns: List[str],p_tb_columns_type: List[str]) -> None:
        # 创建游标
        cur = self.conn.cursor()
        p_tb_column_lines = [column_name + ' ' + column_type for column_name, column_type in zip(p_tb_columns,p_tb_columns_type)]
        sql_str = f"CREATE TABLE {p_tb_name} ({','.join(p_tb_column_lines)})"
        logger.info('sql: ' + sql_str)
        cur.execute(sql_str)

    # 删除表
    def drop_table(self,p_tb_name: str) -> None:
        # 创建游标
        cur = self.conn.cursor()
        sql_str = 'DROP TABLE ' + p_tb_name
        logger.info('sql: ' + sql_str)
        cur.execute(sql_str)

    # 插入数据
    def insert_data(self, p_tb_name: str, p_data: List[Tuple]) -> None:
        # 创建游标
        cur = self.conn.cursor()
        sql_str = f"INSERT INTO {p_tb_name} VALUES({','.join(['?'] * len(p_data[0]))})"
        logger.info('sql: ' + sql_str)
        cur.executemany(sql_str,p_data)


    # 查询数据
    def select_data(self, p_sql_str: str) -> List[Tuple]:
        # 创建游标
        cur = self.conn.cursor()
        logger.info('sql: ' + p_sql_str)
        cur.execute(p_sql_str)
        sql_answer = cur.fetchall()
        if len(sql_answer) > 10:
            raise ValueError("too many query results")
        return sql_answer
    
def main():
    sqldb = Sqldb(DB_FILE_PATH)
    
    result = sqldb.select_data("select 股票代码 ,round(([收盘价(元)] -[昨收盘(元)])/[昨收盘(元)]*100,2)||'%' from A股票日行情表 where 交易日 = '20210105' and 股票代码 in (select 股票代码 from A股公司行业划分表 where 行业划分标准 = '中信行业分类' and 一级行业名称 = '综合金融' and 交易日期 = '20210105') order by ([收盘价(元)] -[昨收盘(元)])/[昨收盘(元)] desc limit 1   ")
    print(post_process_answer("请帮我计算，在20210105，中信行业分类划分的一级行业为综合金融行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。", str(result)))
if __name__ == '__main__':
    main()



        