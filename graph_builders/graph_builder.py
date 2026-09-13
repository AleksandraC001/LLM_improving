import abc
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from model import Model, llama_params1

@dataclass
class Graph:
    pipeline_name: str
    model_name: str
    graph: CompiledStateGraph[Any, None, Any, Any]


class GraphBuilder(abc.ABC):
    name: str

    def __init__(self, model: Model):
        self.model = model
        self.llama_params = llama_params1
        self.gemma_max_tokens = 2000

    def get_llm(self) -> ChatOpenAI:
        return {
            Model.GPT_4O_MINI: ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            Model.GPT_4O: ChatOpenAI(model="gpt-4o", temperature=0.0),
            Model.LLAMA: ChatOpenAI(
                model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
                api_key=None,
                base_url="http://localhost:8001/v1",
                temperature=0,
                max_tokens=self.llama_params.max_tokens,
                timeout=self.llama_params.timeout,
                max_retries=self.llama_params.max_retries,
            ),
            Model.GEMMA: ChatOpenAI(
                model="nvidia/Gemma-4-26B-A4B-NVFP4",
                api_key=None,
                base_url="http://localhost:8002/v1",
                temperature=0,
                max_tokens=self.gemma_max_tokens,
                timeout=900.0,
                max_retries=0
            ),
        }[self.model]

    @abc.abstractmethod
    def build_graph(self) -> CompiledStateGraph[Any, None, Any, Any]:
        pass

    def build(self) -> Graph:
        return Graph(self.name, self.model, self.build_graph())