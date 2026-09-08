import os
import json
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
import prompts_used
from asy_check import problem_text


def load_math_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            problem_text = data['problem']
            doc = Document(
                text=problem_text,
                metadata={
                    "solution": data['solution'],
                    "level": data.get('level', 'Unknown'),
                    "type": data.get('type', 'Unknown'),
                },
                excluded_embed_metadata_keys=["solution", "level", "type"],
                excluded_llm_metadata_keys=["solution", "level", "type"]
            )
            documents.append(doc)
    return documents


def initialize_retriever(train_path='/home/olacz/Downloads/MATH/train/', persist_dir="./math_index2"):
    try:
        topics = os.listdir(train_path)
        print("Files and directories in '", train_path, "' :")
        print(topics)

        documents = []
        for topic in topics:
            documents.extend(load_math_documents(os.path.join(train_path, topic)))
    except Exception as e:
        print(f"Błąd ładowania z katalogu train: {e}")
        documents = []

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_dir):
        print("Ładowanie istniejącego indeksu...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("Tworzenie nowego indeksu...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)

    return VectorIndexRetriever(index=index, similarity_top_k=3)


RETRIEVER = initialize_retriever()

llm_RAG = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    timeout=70.0,
    max_retries=2
)

class RAGEvaluation(BaseModel):
    useful_example_numbers: List[int] = Field(
        description="A list of numbers of the useful examples (e.g., [1, 2]). Return an empty list [] if none are logically useful."
    )

async def rag_agent(state: dict):
    original_problem = state["messages"][0].content

    retrieved_docs = await RETRIEVER.aretrieve(original_problem)

    examples_for_evaluation = ""
    for i, doc in enumerate(retrieved_docs, 1):
        examples_for_evaluation += f"--- Example {i} ---\n"
        examples_for_evaluation += f"Problem: {doc.text}\n\n"

    #print("zadania wybrane przez agenta RAG:")
    #print(examples_for_evaluation)

    prompt = prompts_used.get_rag_eval_prompt(
        original_problem=original_problem,
        rag_found=examples_for_evaluation
    )

    structured_llm = llm_RAG.with_structured_output(RAGEvaluation)
    result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    selected_nums = result.useful_example_numbers
    print("Logicznie dopasowane zadania nr:")
    print(selected_nums)

    context_to_inject = ""
    if selected_nums:
        #print(f"RAG AGENT: Sukces! Wybrano logicznie przydatne przykłady: {selected_nums}.")
        print("Original problem: \n")
        print(original_problem)
        for num in selected_nums:
            if 1 <= num <= len(retrieved_docs):
                selected_idx = num - 1
                selected_doc = retrieved_docs[selected_idx]

                solution = selected_doc.metadata.get("solution", "Brak rozwiązania")
                context_to_inject += (
                    f"--- USEFUL EXAMPLE ---\n"
                    f"Problem: {selected_doc.text}\n"
                    f"Solution: {solution}\n\n"
                )
                print(f"\nWybór agenta RAG nr {selected_idx}:\n__________________________________________________\n")
                print(f"polecenie: {selected_doc.text}")
                print(f"rozwiązanie: {solution}")
                print(f"\n\n__________________________________________________\n")
            else:
                print(f"RAG AGENT: Ostrzeżenie! LLM podał numer spoza zakresu: {num}")
    # else:
    #     print("RAG AGENT: Brak logicznego dopasowania (Zwrócono pustą listę). Odrzucam przykłady.")


    return {"rag_context": context_to_inject}