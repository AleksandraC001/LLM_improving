from typing import Annotated

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import prompts_used
from graph_builders.graph_builder import GraphBuilder
from rag import rag_agent


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


class Builder(GraphBuilder):
    name = 'rag'

    def get_solver(self):
        llm = self.get_llm()

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

        return solver

    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("rag_agent", rag_agent)
        graph_builder.add_node("solver", self.get_solver())

        graph_builder.add_edge(START, "rag_agent")
        graph_builder.add_edge("rag_agent", "solver")
        graph_builder.add_edge("solver", END)

        graph = graph_builder.compile()

        try:
            print("Generowanie wizualizacji grafu...")
            graph_image = graph.get_graph().draw_mermaid_png()
            with open("../podstawa_graph.png", "wb") as f:
                f.write(graph_image)
            print("Graf został zapisany jako 'podstawa_graph.png'")
        except Exception:
            print("Nie udało się wygenerować grafu")

        return graph
