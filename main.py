import os

with open("API_Langsmith", "r") as f:
    API_Langsmith = f.read().strip()
# os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "A_solver_llamma_MATH490"  # only_solver_MATH490
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = API_Langsmith

with open("API_OPEN_AI", "r") as f:
    OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import asyncio
from enum import StrEnum, auto

from pick import pick

from graph_builders import BaselineBuilder, GraphBuilder, McpBuilder, MultiAgentBuilder, RagBuilder, RagWithMcpBuilder
from tests import Evaluator
from dataset import Dataset
from model import Model

if not os.path.exists("API_brave_search") or os.path.getsize("API_brave_search") == 0:
    api_brave = input("Wprowadź API Brave Search:")
    with open("API_brave_search", "w") as f:
        f.write(api_brave)
if not os.path.exists("API_wolf") or os.path.getsize("API_wolf") == 0:
    api_wolfram = input("Wprowadź Wolfram Alpha:")
    with open("API_wolf", "w") as f:
        f.write(api_wolfram)
if not os.path.exists("API_OPEN_AI") or os.path.getsize("API_OPEN_AI") == 0:
    api_openai = input("Wprowadź API OPEN AI:")
    with open("API_OPEN_AI", "w") as f:
        f.write(api_openai)
if not os.path.exists("API_Langsmith") or os.path.getsize("API_Langsmith") == 0:
    api_langsmith = input("Wprowadź API Langsmith:")
    with open("API_Langsmith", "w") as f:
        f.write(api_langsmith)


class Pipeline(StrEnum):
    BASELINE = auto()
    RAG = auto()
    MCP = auto()
    RAG_WITH_MCP = auto()
    MULTI_AGENT = auto()


# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=Model, choices=list(Model), required=True)
# parser.add_argument("--dataset", type=str, required=True)
# args = parser.parse_args()
# model = args.model
# dataset = args.dataset

pipeline_to_builder_module = {
    Pipeline.BASELINE: BaselineBuilder,
    Pipeline.RAG: RagBuilder,
    Pipeline.MCP: McpBuilder,
    Pipeline.RAG_WITH_MCP: RagWithMcpBuilder,
    Pipeline.MULTI_AGENT: MultiAgentBuilder,
}

if __name__ == '__main__':
    title = 'Wybierz z listy rzepływ do rozwiązania zadania:'
    pipeline, _ = pick(list(Pipeline), title)

    title = 'Wybierz zbiór danych do ewaluacji:'
    dataset, _ = pick(list(Dataset), title)

    title = 'Wybierz numer modelu solvera:'
    model, _ = pick(list(Model), title)

    print(f"Rozpoczynam ewaluację {pipeline.name} (Model: {model.name}) na zbiorze {dataset}...")
    graph_builder = pipeline_to_builder_module[pipeline](model)
    evaluator = Evaluator(graph_builder.build())
    asyncio.run(evaluator.evaluate_dataset(dataset, verifier=False))
