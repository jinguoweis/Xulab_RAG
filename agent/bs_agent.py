from copy import deepcopy
from typing import Dict, List, Optional, Union
import re
import sys
# sys.path.append('/home/extra1T/jin/app')
from model.llm import LLM
from .output_parser import BSOutputParser
from .output_wrapper import display
from prompt import BSPromptGenerator
from model.vec_model.vec_model import VectorizeModel
import json
import logging
# from ..retrieve import ToolRetrieval
from chains import DocumentRetrieveChain, DatabaseQueryChain, SqlGeneratorChain, AMAPWeather
# 构建Agent，需要传入llm，工具配置config以及工具检索

data = {'amap_weather':'{"name":"amap_weather","description":"获取对应城市的天气数据"}',
        'RenewInstance':'{"name":"RenewInstance","description":"续费一台包年包月ECS实例"}',
        'amap_poi_query':'{"name":"amap_poi_query","description":"获取对应地区地点的poi信息,你要把把我的地区地点名称连在一起输出作为一个参数,不能让他们之间用逗号分开。比如:问题是帮我查看杭州市的海底捞poi,你需要解析出来的参数是<杭州市海底捞>的参数，是连在一起的"}',
        'Document1':'{"name":"Document1","description":"关于个人所得税的相关问题，税务征收问题以及和税务政策等相关的税务知识问题调用该工具"}'}

DEFAULT_TOOL_LIST = {"Document1": DocumentRetrieveChain(),
                     "SqlGenerator": SqlGeneratorChain(),
                        "DatabaseQuery": DatabaseQueryChain(),
                        "amap_weather": AMAPWeather()}

class BSAgentExecutor:

    def __init__(self,
                 llm: LLM,
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

        self.llm = llm
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
        
        logger = logging.getLogger(__name__)
        
        idx = 0
        final_res = []

        addtional_tool={}
        cx = None

        self.reset()
        self.agent_state["用户问题"] = user_input
        print(user_input)
        # self.prompt_generator.init_plan_prompt(user_input)
        self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
        vec_model = VectorizeModel('/home/extra1T/model_embeding/simcse-chinese-roberta-wwm-ext')
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
        if max_score>0.45:
            addtional_tool[max_key]=DEFAULT_TOOL_LIST[max_key]
            self.prompt_generator.init_task_prompt111(user_input, addtional_tool.values())  ##初始化 self.system_prompt1
            prompt111 = self.prompt_generator.generate1(self.task_no)
            cx =True
        else :
            result = self.llm.generate(user_input)
            return result

        while True:
            idx += 1
            # generate prompt and call llm
            llm_result, exec_result = '', {}
            prompt = self.prompt_generator.generate(self.task_no)
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
            # parse and get tool name and arguments
            action, action_args = self.output_parser.parse_response(llm_result)
            if print_info:
                print(f'|action: {action}, action_args: {action_args}')
            final_res.append(llm_result)
            if action is None:
                # in chat mode, the final result of last instructions should be updated to prompt history
                #带有历史的回复 后续开一个 带历史的 大模型端口
                # global_history = load_history_from_file('/home/workspace/NLP/jinguowei2/app/chains/history.json')
                # result,global_history = llm.sql_generate(user_input,global_history)
                # save_history_to_file(global_history, '/home/workspace/NLP/jinguowei2/app/chains/history.json')
                # return {"result": result,"history":global_history}
                # return {'result': f'模型的回答为:{result}，用户对话历史为:{global_history}'}
                # result = run_client(self.config["llm_url"],user_input)
                # return result
                pass
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
                tool = self.available_tool_list[action]
                try:
                    exec_result = tool(user_input,**action_args)
                    if print_info:
                        print(f'|exec_result: {exec_result}')
                        logger.info(f'|exec_result: {exec_result}')
                    # parse exec result and store result to agent state
                    if exec_result.get("error") is not None:
                        final_res.append(exec_result)
                        if self.error_nums < 3:
                            self.error_nums += 1
                            #self.prompt_generator.init_plan_prompt(user_input)  #继续 初始化 从头再来一遍直到没错 如果大于三次还是错误的 那就直接 返回了
                            #最后的 问题对应的answer也是抛出的异常 
                            self.prompt_generator.init_task_prompt(user_input, self.available_tool_list.values())
                            self.task_no = None
                            continue
                        return final_res
                    else:
                        return exec_result
                    # final_res.append(exec_result.get('result', ''))
                    # self.parse_exec_result(exec_result)
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
                answer = self.llm.generate(user_input)
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
        self.task_no = None
        self.exec_result = ''
        self.agent_state = dict()
        self.error_nums = 0

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
