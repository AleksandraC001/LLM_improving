import asyncio
from typing import Annotated
import os
import argparse

with open("API_Langsmith", "r") as f:
    API_Langsmith = f.read().strip()
#os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "A_test_solver_RAG_490MATH_Llama"
os.environ["LANGSMITH_ENDPOINT"]="https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = API_Langsmith

import httpx
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import prompts_used
import tests

with open("API_OPEN_AI", "r") as f:
    OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    calls: Annotated[int, lambda x, y: x + y]
    rag_context: str | None
    feedback: str | None
    correct: str
    iterations: int
    to_end: bool
    system_prompt_text: str
    rag_outputs_count: int


async def solver(state: State):
    messages = state["messages"]
    rag_context = state.get("rag_context")
    system_prompt_text = prompts_used.get_solver_RAG_prompt(rag_context=rag_context)
    feedback = state.get("feedback")

    if feedback:
        system_prompt_text = system_prompt_text + feedback

    system_prompt = SystemMessage(content=system_prompt_text)
    prompt_with_history = [system_prompt] + messages
    response = await llm.ainvoke(prompt_with_history)
    return {"messages": [response], "system_prompt_text": system_prompt}

from rag import rag_agent

graph_builder = StateGraph(State)
graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("solver", solver)

graph_builder.add_edge(START, "rag_agent")
graph_builder.add_edge("rag_agent", "solver")
graph_builder.add_edge("solver", END)

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
        print("wstęp do algebry:")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/prealgebra", "Level 5", graph, verifier=False)
        print("algebra: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/algebra", "Level 5", graph, verifier=False)
        print("wstęp do analizy matematycznej: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/precalculus", "Level 5", graph, verifier=False)
        print("kombinatoryka i prawdopodobieństwo: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/counting_and_probability", "Level 5", graph, verifier=False)
        print("teoria liczb: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/number_theory", "Level 5", graph, verifier=False)
        print("średnio-zaawansowana algebra: ")
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/intermediate_algebra", "Level 5", graph, verifier=False)

asyncio.run(main())

#dalej zobaczyć przepływ tych testów i jak sie raporty zapisują, czy werdykt llm dać tą odpowiedź całą jego czy TAK NIE