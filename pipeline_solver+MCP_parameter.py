import os

with open("API_Langsmith", "r") as f:
    API_Langsmith = f.read().strip()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "testowy"
os.environ["LANGSMITH_ENDPOINT"]="https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = API_Langsmith

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import prompts_used
import tests
from tools_MCP_async import tools_required, tools_auto

from langchain_openai import ChatOpenAI
import asyncio

import argparse

with open("API_OPEN_AI", "r") as f:
    OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--parameter", type=str, required=True)

args = parser.parse_args()
model = args.model
dataset = args.dataset
parameter_choice = args.parameter

if model == "gpt-4o-mini":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
elif model == "gpt-4o":
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
elif model == "llama-3.3-70B-Instruct":
    llm = ChatOpenAI(
        model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
        api_key="empty",
        base_url="http://localhost:8001/v1",
        temperature=0,
        max_tokens=2000,
        timeout=240.0,
        max_retries=2
    )
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    calls: Annotated[int, lambda x, y: x + y]
    rag_context: str | None
    feedback: str | None
    correct: str
    iterations: int
    to_end: bool
    system_prompt_text: str

if parameter_choice == "required":
    tools_parameter = tools_required
if parameter_choice == "auto":
    tools_parameter = tools_auto

llm_with_tools = llm.bind_tools(tools_parameter, parallel_tool_calls=False, tool_choice=parameter_choice)

async def solver(state: State):
    messages = state["messages"]

    if parameter_choice== "required":
        system_prompt = SystemMessage(content=prompts_used.get_solver_MCP_prompt_required())
    elif parameter_choice == "auto":
        system_prompt = SystemMessage(content=prompts_used.get_solver_MCP_prompt_auto())
    prompt_with_history = [system_prompt] + messages
    response = await llm_with_tools.ainvoke(prompt_with_history)
    return {"messages": [response], "system_prompt_text": system_prompt}

def solver_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if parameter_choice== "required" and tool_name == "submit_to_verifier":
            return END
        return "tools"
    return END

tool_node = ToolNode(tools_parameter)
graph_builder = StateGraph(State)

graph_builder.add_node("solver", solver)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "solver")
graph_builder.add_edge("tools", "solver")
graph_builder.add_conditional_edges("solver", solver_router, {"tools": "tools", END: END})

graph = graph_builder.compile()

try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("podstawa_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'podstawa_graph.png'")
except Exception:
    print("Nie udało się wygenerować grafu")

if __name__ == "__main__":
    async def main():
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/prealgebra", "Level 5", graph, verifier=False)
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/algebra", "Level 5", graph, verifier=False)
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/precalculus", "Level 5", graph, verifier=False)
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/counting_and_probability", "Level 5", graph, verifier=False)
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/number_theory", "Level 5", graph, verifier=False)
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/intermediate_algebra", "Level 5", graph, verifier=False)

asyncio.run(main())
