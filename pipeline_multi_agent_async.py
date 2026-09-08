import asyncio

from typing import Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import prompts_used
from rag import rag_agent
import tests


from tools_MCP_async import tools_auto

from langchain_openai import ChatOpenAI


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


llm = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    max_tokens=2000,
    timeout=120.0,
    max_retries=2
)

llm_with_tools = llm.bind_tools(tools_auto, parallel_tool_calls=False, tool_choice="required")
print("Narzędzia załadowane pomyślnie!")


async def solver(state: State):
    messages = state["messages"]
    rag_context = state.get("rag_context")
    system_prompt_text = prompts_used.get_solver_prompt(rag_context=rag_context)
    feedback = state.get("feedback")
    if feedback:
        system_prompt_text = system_prompt_text + feedback
    system_prompt = SystemMessage(content=system_prompt_text)
    prompt_with_history = [system_prompt] + messages

    response = await llm_with_tools.ainvoke(prompt_with_history)
    print("Odpowiedź solvera:")
    print(response)
    return {"messages": [response], "system_prompt_text": system_prompt}


def solver_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "submit_to_verifier":
            print("Router: Solver przekazuje odpowiedź do Weryfikatora.")
            return "verifier"
        print(f"Router: Agent wykonuje obliczenia/szuka w {tool_name}...")
        return "tools"
    return "verifier"

from verifier import verifier_router, verifier
tool_node = ToolNode(tools_auto)
graph_builder = StateGraph(State)

graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("solver", solver)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("verifier", verifier)

graph_builder.add_edge(START, "rag_agent")
graph_builder.add_edge("rag_agent", "solver")
graph_builder.add_conditional_edges("solver", solver_router, {"tools": "tools", "verifier": "verifier"})
graph_builder.add_edge("tools", "solver")
graph_builder.add_conditional_edges("verifier", verifier_router, {"solver": "solver", END: END})

graph = graph_builder.compile()

try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("różne/multi_agent_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'multi_agent_graf.png'")
except Exception:
    print("Nie udało się wygenerować grafu")

if __name__ == "__main__":
    async def main():
        await tests.evaluate_llm("/home/olacz/Downloads/MATH480/test/algebra", "Level 5", graph, verifier=True)
        # await evaluate_llm("/home/olacz/Downloads/MATH480/test/geometry", "Level 5")
        # await evaluate_llm("/home/olacz/Downloads/MATH480/test/number_theory", "Level 3")


    asyncio.run(main())