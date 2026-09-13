from typing import Annotated

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from graph_builders.graph_builder import GraphBuilder
import prompts_used


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    feedback: str | None
    system_prompt_text: str


class Builder(GraphBuilder):
    name = 'baseline'

    def get_solver(self):
        llm = self.get_llm()

        async def solver(state: State) -> dict:
            messages = state["messages"]
            system_prompt = SystemMessage(content=prompts_used.get_baseline_solver_prompt())
            prompt_with_history = [system_prompt] + messages
            response = await llm.ainvoke(prompt_with_history)
            return {"messages": [response], "system_prompt_text": system_prompt}

        return solver

    def build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("solver", self.get_solver())
        graph_builder.add_edge(START, "solver")
        graph_builder.add_edge("solver", END)
        return graph_builder.compile()
