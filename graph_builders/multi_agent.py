from typing import Annotated

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import prompts_used
from graph_builders.graph_builder import GraphBuilder
from model import Model, llama_params2
from rag import rag_agent
from tools_MCP_async import get_tools_auto
from verifier import verifier_router, verifier


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


def solver_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "submit_to_verifier":
            return "verifier"
        return "tools"
    return "verifier"


class Builder(GraphBuilder):
    name = 'multi_agent'

    def __init__(self, model: Model):
        super().__init__(model)
        self.llama_params = llama_params2
        self.gemma_max_tokens = 1200

    def get_solver(self):
        llm = self.get_llm()
        llm_with_tools = llm.bind_tools(get_tools_auto(), parallel_tool_calls=False, tool_choice="auto")
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
            return {"messages": [response], "system_prompt_text": system_prompt}

        return solver

    def build_graph(self):
        tool_node = ToolNode(get_tools_auto())
        graph_builder = StateGraph(State)

        graph_builder.add_node("rag_agent", rag_agent)
        graph_builder.add_node("solver", self.get_solver())
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
            with open("../różne/multi_agent_graph.png", "wb") as f:
                f.write(graph_image)
            print("Graf został zapisany jako 'multi_agent_graf.png'")
        except Exception:
            print("Nie udało się wygenerować grafu")

        return graph
