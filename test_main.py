import asyncio
from itertools import product
import os
with open("API_OPEN_AI") as f:
    OPENAI_API_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from dataset import Dataset
from model import Model
from pipeline import Pipeline, pipeline_to_builder_module
from tests import Evaluator


def execute(pipeline: Pipeline, model: Model, dataset: Dataset):
    print(f"Rozpoczynam ewaluację {pipeline.name} (Model: {model.name}) na zbiorze {dataset}...")
    graph_builder = pipeline_to_builder_module[pipeline](model)
    evaluator = Evaluator(graph_builder.build())
    asyncio.run(evaluator.evaluate_dataset(dataset, verifier=(pipeline == Pipeline.RAG_WITH_MCP_AND_VERIFIER)))

PIPELINES = [
    Pipeline.BASELINE,
    Pipeline.MCP,
    Pipeline.RAG,
    Pipeline.RAG_WITH_MCP,
    Pipeline.RAG_WITH_MCP_AND_VERIFIER,
]

MODELS = [
    Model.GPT_4O_MINI,
    Model.GEMMA,
    Model.LLAMA,
]

DATASETS = [
    Dataset.AIME,
    Dataset.MATH480,
]

if __name__ == "__main__":
    for pipeline, model, dataset in product(PIPELINES, MODELS, DATASETS):
        execute(pipeline, model, dataset)
        execute(pipeline, model, dataset)
        execute(pipeline, model, dataset)

