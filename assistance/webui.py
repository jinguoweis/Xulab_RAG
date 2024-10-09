import gradio as gr
from client import run_client
# # def greet(name):
# #     a1 = name
# #     return a1
# def search_question(query):
#     url = "http://192.168.10.211:9090/searcher"
#     return run_client(url, query) # 单元测试
# def llm_question(query):
#     url = "http://192.168.10.211:9092/llm_model"
#     return run_client(url, query) # 单元测试
def dialogue_question(query):
    url = url = "http://127.0.0.1:9095/dialogue_manager"
    result = run_client(url, query)
    result = result['answer']
    return result# 单元测试
# def dialogue_question(query):
#     url = "http://192.168.0.110:9090/searcher"
#     return run_client(url, query)# 单元测试
# iface1 = gr.Interface(fn=search_question, inputs=gr.Textbox(label="请输入要查询的问题"),outputs=gr.Text())
# iface2 = gr.Interface(fn=llm_question, inputs=gr.Textbox(label="请输入要查询的问题"),outputs=gr.Text())
iface3 = gr.Interface(fn=dialogue_question, inputs=gr.Textbox(label="请输入要查询的问题"),outputs=gr.Text())
# tabbed_interface = gr.TabbedInterface([iface1, iface2,iface3], ["向量库查询", "大模型的原始能力回复","大模型加向量库查询结果回复"])
tabbed_interface = gr.TabbedInterface([iface3], ["Xulab Agent大模型"])
# tabbed_interface = gr.TabbedInterface([iface3], ["税务问答机器人"])
tabbed_interface.launch(share=True)
# import gradio as gr
# #该函数有3个输入参数和2个输出参数
# def greet(name, is_morning, temperature):
#     salutation = "Good morning" if is_morning else "Good evening"
#     greeting = f"{salutation} {name}. It is {temperature} degrees today"
#     celsius = (temperature - 32) * 5 / 9
#     return greeting, round(celsius, 2)
# demo = gr.Interface(
#     fn=greet,
#     #按照处理程序设置输入组件
#     inputs=["text", "checkbox", gr.Slider(0, 100)],
#     #按照处理程序设置输出组件
#     outputs=["text", "number"],
# )
# demo.launch()