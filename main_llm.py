import groq
import httpx
import langgraph.errors
from groq import Groq
from langchain_groq import ChatGroq
import json
import os
from langchain_core.prompts import ChatPromptTemplate
import re
from sympy.parsing.latex import parse_latex
import antlr4
from importlib.metadata import version
from sympy.external import import_module
from sympy.parsing.latex.errors import LaTeXParsingError
import prompts
from preprocessing import *
from typing import Literal
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langchain_community.tools import WolframAlphaQueryRun
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# antlr4 = import_module('antlr4')
#
# if None in [antlr4, MathErrorListener] or \
#         not version('antlr4-python3-runtime').startswith('4.11'):
#     raise ImportError("LaTeX parsing requires the antlr4 Python package,"
#                       " provided by pip (antlr4-python3-runtime) or"
#                       " conda (antlr-python-runtime), version 4.11")

path = '/home/olacz/Downloads/MATH/train/'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")

print(dir_list)

with open("groq_api_key", "r") as f: GROQ_API_KEY = f.read()

MODEL = 'llama-3.3-70b-versatile'

llm = ChatGroq(
    model_name=MODEL,
    temperature=0.7,
    api_key=GROQ_API_KEY,
)
#duck duck go:
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

with open("duck_API", "r") as f: DUCK_API_KEY = f.read()

duck_duck_go_tool = DuckDuckGoSearchRun()

# duckduckgo_tool = Tool(
#     name="DuckDuckGo Search",
#     func=search.run()
#     description="Search for information in the internet. Use if you need additional explaining to the mathematical problem"
# )

with open("wolf_API", "r") as f: WOLF_API_KEY = f.read()
os.environ["WOLFRAM_ALPHA_APPID"] = ""
wolfram_tool = WolframAlphaQueryRun(
    api_wrapper=WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLF_API_KEY)
)

from langchain.agents import initialize_agent, AgentType, create_react_agent, AgentExecutor


from python_executor import PythonExecutor
python_tool = PythonExecutor()

tools = [duck_duck_go_tool, wolfram_tool, python_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=prompts.main_prompt
                    )
                ]
                + state["messages"]
            )
        ]
    }
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()


def evaluate_llm3(dir_, lvl):
    print(f"Evaluating {dir_} on {lvl}")
      # all = 0s 1% of his previous week's weig
    right_raw = 0
    right_after_cleaning = 0
    right_latex = 0
    wrong = 0
    i = 0
    for filename in os.listdir(dir_):
        with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)  # Parsowanie JSON do obiektu Pythona (słownik lub lista)
            if data["level"] != lvl:
                continue
            print(data)
            print(f"{i=}, {right_raw=}, {right_after_cleaning=}, {right_latex=}, {wrong=}")
            i += 1
            x = re.search(r'\\boxed{(.*)}', data["solution"])
            right_answer = x.group(1)
            messages = [HumanMessage(content=data["problem"])]
            try:
                llm_full_answer = agent.invoke({"messages": messages})
            except httpx.ReadTimeout:
                # TODO: implement exponential backoff
                print("Timeout")
                wrong += 1
                continue
            except groq.BadRequestError:
                # TODO: implement exponential backoff
                print("Bad request")
                wrong += 1
                continue
            except langgraph.errors.GraphRecursionError:
                # TODO: implement exponential backoff
                print("Recursion limit exceeded")
                wrong += 1
                continue

            for m in messages:
                m.pretty_print()
            for j, m in enumerate(llm_full_answer['messages'], start=1):
                print(j, m)
            print(llm_full_answer)
            print(list(llm_full_answer.keys()))
            # print(llm_full_answer['right_answer'])
            # print(llm_full_answer['llm_answer'])
            # llm_answer = re.search(r'\\boxed{(.*)}', str(llm_full_answer)).group(1)
            last_message = llm_full_answer['messages'][-1].content
            print(f"OSTATNIA WIADOMOSC: {last_message}")
            try:
                llm_answer = re.search(r'\\boxed{(.*)}', last_message).group(1)
            except AttributeError:
                print(f"Nie znalazł boxed w {last_message}.")
                wrong += 1
                continue
            print(f"{llm_answer=}")
            # llm_answer = re.search(r'\\boxed{(.*)}', str(llm_full_answer)).group(1)
            if right_answer == llm_answer:
                right_raw += 1
                continue

            if is_equiv(right_answer, llm_answer):
                right_after_cleaning += 1
                continue

            try:
                parsed_right_answer = parse_latex(right_answer)
                parsed_llm_answer = parse_latex(llm_answer)
                if parsed_right_answer == parsed_llm_answer:
                    right_latex += 1
                    continue
            except:
                pass

            wrong += 1
            print(f"{right_answer=}, {llm_answer=}")
            if i > 200:
                break


#evaluate_llm3("/home/olacz/Downloads/MATH/test/number_theory", "Level 5")

#evaluate_llm3("/home/olacz/Downloads/MATH/test/geometry", "Level 5")

#evaluate_llm3("/home/olacz/Downloads/MATH/test/counting_and_probability", "Level 5")
#evaluate_llm3("/home/olacz/Downloads/MATH/test/precalculus", "Level 5")
evaluate_llm3("/home/olacz/Downloads/MATH/test/prealgebra", "Level 5")

