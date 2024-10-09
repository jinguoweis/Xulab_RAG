from openai import OpenAI
import csv
import pandas as pd

client = OpenAI(
    api_key="sk-rLAcY2kNadfjjgh2MSrOa2UAHaho12LYpoxj562vOaEQLV0C",
    base_url="https://api.chatanywhere.tech"
 # base_url="https://api.chatanywhere.org/v1"
)
# 非流式响应
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return completion.choices[0].message.content

def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
# 调用接口进行测试 
# if __name__ == '__main__':
#     messages = [{'role': 'user','content': '你是一个问题改写助手，我给你提供一个问题，帮我改写这个问题，给我提供三个同义的问题版本。问题如下：泰国北部南邦省洪水致一人失踪，数千居民被迫撤离。洪水发生的主要原因是什么？'},]
#     # 非流式调用
#     # gpt_35_api(messages)
#     # 流式调用
#     # print(type(messages))
#     cc = gpt_35_api(messages)
#     print(cc)
# 读取输入的 CSV 文件
input_file = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/test_question_query_rewrite.csv'
df = pd.read_csv(input_file)
input_file1 = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/test_question_all.csv'
df1 = pd.read_csv(input_file1)
# 创建输出 CSV 文件路径
output_file = '/home/extra1T/jin/GoMate/RAG/retrieve/RAG/data/output_answers_query_rewrite.csv'

# 打开输出 CSV 文件，准备写入
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    
    # 写入表头
    writer.writerow(['question'])  # 只写一个 'question' 列
    
    # 遍历每一行问题
    for row in range(265,len(df)):
        question = df['question'][row]
        question1 = df1['question'][row]
        messages = [{'role': 'user', 'content': question}]
        
        # 调用 GPT-3.5 API 获取回答，假设返回带序号的多条回答列表
        answers = gpt_35_api(messages)  # 返回列表类型
        answers = question1+answers
        # 检查返回结果是否为列表，确保其有效性
        # if not isinstance(answers, list):
        #     print(f"Warning: Unexpected answer format for question: {question}")
        #     continue  # 如果不是列表，则跳过该行
        
        # 将列表中的回答用空格连接为字符串
        # combined_answers = ' '.join(answers)
        
        # 将问题和答案组合在一起
        combined_result = answers
        
        # 将组合后的字符串写入 CSV 文件
        writer.writerow([combined_result])
        # print(f"Processed: {combined_result}")

#对保存好的 数据 进行 处理 
# import pandas as pd

# # 读取 CSV 文件
# input_file = '/home/extra1T/jin/GoMate/output_answers2.csv'
# df = pd.read_csv(input_file)

# # 去掉每一行的换行符
# df = df.applymap(lambda x: x.replace('\n', '') if isinstance(x, str) else x)

# # 覆盖原始 CSV 文件
# df.to_csv(input_file, index=False, encoding='utf-8')

# print("换行符已移除，原始 CSV 文件已被覆盖。")