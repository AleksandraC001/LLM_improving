import os
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

with open("API_Langsmith", "r") as f:
    API_Langsmith = f.read().strip()
#os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "A_solver_llamma_MATH490"#only_solver_MATH490
os.environ["LANGSMITH_ENDPOINT"]="https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = API_Langsmith

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import prompts_used
import tests

with open("API_OPEN_AI", "r") as f:
    OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
model = args.model
dataset = args.dataset

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
        timeout=900.0,
        max_retries=0
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    feedback: str | None
    system_prompt_text: str

with open("API_OPEN_AI", "r") as f:
    OPENAI_API_KEY = f.read().strip()

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

async def solver(state: State):
    messages = state["messages"]
    system_prompt = SystemMessage(content=prompts_used.get_baseline_solver_prompt())
    prompt_with_history = [system_prompt] + messages
    response = await llm.ainvoke(prompt_with_history)
    return {"messages": [response], "system_prompt_text": system_prompt}

graph_builder = StateGraph(State)
graph_builder.add_node("solver", solver)
graph_builder.add_edge(START, "solver")
graph_builder.add_edge("solver", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    async def main():
        print("wstęp do algebry:")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/prealgebra", "Level 5", graph, verifier=False)
        print("algebra: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/algebra", "Level 5", graph, verifier=False)
        print("wstęp do analizy matematycznej: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/precalculus", "Level 5", graph, verifier=False)
        print("kombinatoryka i prawdopodobieństwo: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/counting_and_probability", "Level 5", graph,
                                 verifier=False)
        print("teoria liczb: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/number_theory", "Level 5", graph, verifier=False)
        print("średnio-zaawansowana algebra: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/intermediate_algebra", "Level 5", graph,
                                 verifier=False)

asyncio.run(main())
