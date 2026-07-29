import os
import json
import asyncio
from pprint import pprint
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import prompts_used

from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document

from collections import Counter
import httpx
from langgraph.errors import GraphRecursionError

from enum import Enum
from MCP_tools_async import tools

from langchain_openai import ChatOpenAI
import openai

def create_conversation(messages):
    conversation_transcript = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_transcript += f"USER QUESTION: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                conversation_transcript += f"SOLVER (TOOL ATTEMPT):\n{msg.tool_calls}\n"
            else:
                conversation_transcript += f"SOLVER: {msg.content}\n\n"
        elif isinstance(msg, ToolMessage):
            conversation_transcript += f"SYSTEM (OUTPUT): {msg.content}\n\n"
    return conversation_transcript


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


# ___________________ RAG ____________________
path = '/home/olacz/Downloads/MATH/train/'
try:
    topics = os.listdir(path)
    print("Files and directories in '", path, "' :")
    print(topics)

    documents = []
    for topic in topics:
        documents.extend(load_math_documents(os.path.join(path, topic)))
except Exception as e:
    print(f"Błąd ładowania z katalogu train: {e}")
    documents = []

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_dir = "./math_index2"
if os.path.exists(persist_dir):
    print("Ładowanie istniejącego indeksu...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
else:
    print("Tworzenie nowego indeksu...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)


class ModelEvaluation(Enum):
    ERROR = 0
    WRONG = 1
    RIGHT = 2


async def evaluate_model_on_problem(data, semaphore: asyncio.Semaphore, stats_counter: Counter) -> ModelEvaluation:
    async with semaphore:
        inputs = {"messages": [HumanMessage(content=data["problem"])]}
        try:
            print(
                f"\n\n___________________________________\nNowe zadanie: {data['problem']}\n____________________________________\n")
            result = await graph.ainvoke({"messages": [{"role": "user", "content": data["problem"]}]},
                                         config={"recursion_limit": 45})
            for i in result["messages"]:
                pprint(i)
            print(result["messages"][-1].content)
            print('\n')
        except httpx.ReadTimeout:
            print("Timeout")
            return ModelEvaluation.ERROR
        except openai.APITimeoutError:
            print("Timeout - Endpoint API nie odpowiedział w wyznaczonym czasie")
            return ModelEvaluation.ERROR
        except GraphRecursionError:
            print("Recursion limit exceeded (Model się zapętlił)")
            return ModelEvaluation.ERROR
        except openai.BadRequestError as e:
            print(f"Bad request - SZCZEGÓŁY BŁĘDU: {e}")
            return ModelEvaluation.ERROR
        except Exception as e:
            print(f"Inny błąd: {e}")
            return ModelEvaluation.ERROR

        last_messages_without_verifier = result["messages"][-3:-1]

        eval_prompt = f"""You are an expert math grader. Compare the correct answer and the student's answer.

Correct solution/answer: {data["solution"]}
Student's solution last steps: ...{last_messages_without_verifier}

Instructions:
1. Identify the final conclusion in the correct solution (it might be inside \\boxed{{}}).
2. Identify the final answer in the student's text. You can look at their last steps to locate their final conclusion, but DO NOT grade the steps.
3. Check if these two final answers are mathematically equivalent. Ignore differences in formatting, LaTeX syntax, and fractions vs decimals.
4. Briefly explain your reasoning in 1-2 sentences.
5. End your response with exactly "Verdict: YES" or "Verdict: NO".
"""
        eval_response = await llm.ainvoke([HumanMessage(content=eval_prompt)])
        llm_decision = eval_response.content.strip().upper()

        evaluation = ModelEvaluation.ERROR
        if "YES" in llm_decision:
            print(f"Sukces! LLM uznał odpowiedzi za równoważne")
            print(f"odpowiedź systemu: {last_messages_without_verifier}, \n odpowiedź oczekiwana: {data['solution']}")
            print(f"uzasadnienie: {llm_decision}")
            evaluation = ModelEvaluation.RIGHT
        elif "NO" in llm_decision:
            print(
                f"odpowiedź systemu: {last_messages_without_verifier}, \nniezgodna z odpowiedzią oczekiwaną: {data['solution']}")
            print(f"Powód: {llm_decision}")
            evaluation = ModelEvaluation.WRONG
        else:
            print(f"w ewaluacji przez LLM zabrakło werdyktu")

        stats_counter[evaluation] += 1
        print(f"Aktualne statystyki: {', '.join(f'{k.name}: {v}' for k, v in stats_counter.items())}\n"
              f"#####################################################################################")
        return evaluation


async def evaluate_llm(dir_, lvl):
    print(f"Evaluating {dir_} on {lvl}")

    MAX_CONCURRENT_TASKS = 5
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    stats_counter = Counter()
    tasks = []

    try:
        filenames = sorted(os.listdir(dir_))
    except FileNotFoundError:
        print(f"Brak katalogu {dir_}")
        return

    for i, filename in enumerate(filenames):
        if len(tasks) >= 40:
            break
        with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
            if data["level"] != lvl:
                continue

            task = evaluate_model_on_problem(data, semaphore, stats_counter)
            tasks.append(task)

    print(f"Znaleziono {len(tasks)} zadań do przetworzenia.")

    await asyncio.gather(*tasks)
    print("\n--- Zakończono ewaluację ---")
    print(f"Końcowe statystyki: {', '.join(f'{k.name}: {v}' for k, v in stats_counter.items())}")


# SEKCJA 4: LOGIKA AGENTA (LANGGRAPH)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    calls: Annotated[int, lambda x, y: x + y]
    rag_context: str | None
    feedback: str | None
    correct: str
    iterations: int
    to_end: bool


llm = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    max_tokens=2000,
    timeout=120.0,
    max_retries=2
)

llm_RAG = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    timeout=70.0,
    max_retries=2
)

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False, tool_choice="required")
print("Narzędzia załadowane pomyślnie!")


class RAGEvaluation(BaseModel):
    useful_example_numbers: List[int] = Field(
        description="A list of numbers of the useful examples (e.g., [1, 2]). Return an empty list [] if none are logically useful."
    )

async def rag_agent(state: State):
    print("\n--- RAG AGENT: Szukam i oceniam podobne zadania ---")
    original_problem = state["messages"][0].content

    retrieved_docs = await retriever.aretrieve(original_problem)

    examples_for_evaluation = ""
    for i, doc in enumerate(retrieved_docs, 1):
        examples_for_evaluation += f"--- Example {i} ---\n"
        examples_for_evaluation += f"Problem: {doc.text}\n\n"

    print("zadania wybrane przez agenta RAG:")
    print(examples_for_evaluation)

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
    if not selected_nums:
        print("RAG AGENT: Brak logicznego dopasowania (Zwrócono pustą listę). Odrzucam przykłady.")
    else:
        print(f"RAG AGENT: Sukces! Wybrano logicznie przydatne przykłady: {selected_nums}.")
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
            else:
                print(f"RAG AGENT: Ostrzeżenie! LLM podał numer spoza zakresu: {num}")

    return {"rag_context": context_to_inject}


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
    return {"messages": [response]}


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


class NewVerificationResult(BaseModel):
    is_correct: bool = Field(
        description="True ONLY if the solution perfectly meets all 5 criteria. False if there is ANY flaw, hallucination, or unhandled tool error.")
    answer: str = Field(
        description="If is_correct is True: provide the exact final answer from the solution in \\boxed{final_answer} format. If is_correct is False: return an empty string ''.")
    feedback: str = Field(
        description="If is_correct is False: State the exact fail reason and provide a clear instruction for Agent Solver on what to fix. If is_correct is True: return an empty string ''.")


async def verifier(state: State):
    print("--- WERYFIKATOR ---")
    new_iteration = state.get("iterations", 0) + 1
    if new_iteration >= 3:
        to_end_flag = True
        msg = [SystemMessage(content="Verification limit")]
        return {"messages": msg, "to_end": to_end_flag, "iterations": new_iteration}

    print(f"Rozpoczynam {new_iteration} iterację weryfikatora")
    messages = state["messages"]
    verifier_prompt = prompts_used.new_get_verifier_prompt(create_conversation(messages))
    structured_llm = llm.with_structured_output(NewVerificationResult)

    try:
        print("Weryfikator myśli...\n")
        response = await structured_llm.ainvoke([HumanMessage(content=verifier_prompt)])
        print("WERYFIKATOR:\n")
        print(response)
        to_end_flag = response.is_correct
        if response.is_correct:
            answer = response.answer
        else:
            answer = response.feedback
    except (httpx.ReadTimeout, openai.APITimeoutError):
        print("\n!!! WERYFIKATOR TIMEOUT: Zwracam sztuczny komunikat błędu !!!\n")
        answer = "VERDICT: FEEDBACK: AWARIA WERYFIKACJI - Weryfikator uległ awarii z powodu zbyt długiego czasu oczekiwania na odpowiedź (Timeout)."
        to_end_flag = True

    return {"messages": answer, "to_end": to_end_flag, "iterations": new_iteration}


def verifier_router(state: State):
    return END if state["to_end"] else "solver"


# Graf
tool_node = ToolNode(tools)
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
    with open("multi_agent_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'multi_agent_graf.png'")
except Exception:
    print("Nie udało się wygenerować grafu")

if __name__ == "__main__":
    async def main():
        await evaluate_llm("/home/olacz/Downloads/MATH/test/algebra", "Level 5")
        # await evaluate_llm("/home/olacz/Downloads/MATH/test/geometry", "Level 5")
        # await evaluate_llm("/home/olacz/Downloads/MATH/test/number_theory", "Level 3")


    asyncio.run(main())