# MCP + RAG + weryfikator LLM-as-a-Judge, ewaluacja programu LLM-as-a Judge
import os
import json
from pprint import pprint
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from sympy import sympify

import preprocessing_2
import prompts_used

from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core.schema import Document

from collections import Counter

import re
import httpx
from langgraph.errors import GraphRecursionError

from enum import Enum
from MCP_tools import tools

# #gdy nvidia jest zajęta
# with open("groq_api_key", "r") as f:
#     GROQ_API_KEY = f.read().strip()

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

#ładowanie dokumentów za zbioru math - w metadanych stopień trudności, rozwiązanie i typ zadania
def load_math_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            problem_text = data['problem']
            #content = f"Problem:\n{data['problem']}\n\nSolution:\n{data['solution']}"
            doc = Document(
                text=problem_text,
                metadata={
                    "solution": data['solution'],
                    "level": data.get('level', 'Unknown'),
                    "type": data.get('type', 'Unknown'),
                },
                excluded_embed_metadata_keys=["solution", "level", "type"],
                excluded_llm_metadata_keys = ["solution", "level", "type"]
            )
            documents.append(doc)
    return documents

#___________________RAG____________________
path = '/home/olacz/Downloads/MATH/train/'
topics = os.listdir(path)
print("Files and directories in '", path, "' :")
print(topics)

documents = []
for topic in topics:
    documents.extend(load_math_documents(os.path.join(path, topic)))

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

def evaluate_model_on_problem(data) -> ModelEvaluation:
    inputs = {"messages": [HumanMessage(content=data["problem"])]}
    try:
        print(f"________________________________\nNowe zadanie: {data['problem']}\n_________________________________\n")
        result = graph.invoke({"messages": [{"role": "user", "content": data["problem"]}]},
                              config={"recursion_limit": 26})
        for i in result["messages"]:
            pprint(i)
        print(result["messages"][-1].content)
        print('\n')
    except httpx.ReadTimeout:
        print("Timeout")
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
    #last_message = result['messages'][-1].content
    eval_prompt = f"""You are an expert math grader. Compare the correct answer and the student's answer.

Correct solution/answer: {data["solution"]}
Student's solution last steps: ...{last_messages_without_verifier}

Instructions:
1. Identify the final conclusion in the correct solution (it might be inside \boxed{{}}).
2. Identify the final answer in the student's text. You can look at their last steps to locate their final conclusion, but DO NOT grade the steps.
3. Check if these two final answers are mathematically equivalent. Ignore differences in formatting, LaTeX syntax, and fractions vs decimals.
4. Briefly explain your reasoning in 1-2 sentences.
5. End your response with exactly "Verdict: YES" or "Verdict: NO".
"""
    eval_response = llm.invoke([HumanMessage(content=eval_prompt)])
    llm_decision = eval_response.content.strip().upper()
    if "YES" in llm_decision:
        print(f"Sukces! LLM uznał odpowiedzi za równoważne")
        print(f"odpowiedź systemu: {last_messages_without_verifier}, \n odpowiedź oczekiwana: {data['solution']}")
        print(f"uzasadnienie: {llm_decision}")
        return ModelEvaluation.RIGHT
    if "NO" in llm_decision:
        print(f"odpowiedź systemu: {last_messages_without_verifier}, \nniezgodna z odpowiedzią oczekiwaną: {data['solution']}")
        print(f"Powód: {llm_decision}")
        return ModelEvaluation.WRONG
    print(f"w ewaluacji przez LLM zabrakło werdyktu")
    return ModelEvaluation.ERROR


def evaluate_llm(dir_, lvl):
    print(f"Evaluating {dir_} on {lvl}")
    i = 0
    c = Counter()
    for filename in sorted(os.listdir(dir_)): #dzięki sorted mamy zadania w tej samej kolejności zawsze
        if i > 20:
            break
        with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)  # Parsowanie JSON do obiektu Pythona (słownik lub lista)
            if data["level"] != lvl:
                continue
            model_evaluation_on_problem = evaluate_model_on_problem(data)
            c[model_evaluation_on_problem] += 1
            i += 1
            print(f"aktualny stan: {i=}, statystyki: {', '.join(f'{k.name}: {v}' for k, v in c.items())}\n\n"
                  f"#####################################################################################")


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

from langchain_openai import ChatOpenAI
import openai
from langchain.chat_models import init_chat_model

# llm = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider="groq",
#     temperature=0,
#     api_key=GROQ_API_KEY,
#     max_tokens=2000,  # chroni przed ucięciem długich rozwiązań matematycznych
#     timeout=120.0,
#     max_retries=2
# )
#
# llm_RAG = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider="groq",
#     temperature=0,
#     api_key=GROQ_API_KEY,
#     max_tokens=2000,  # chroni przed ucięciem długich rozwiązań matematycznych
#     timeout=70.0,
#     max_retries=2
# )

llm = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    max_tokens=2000,  # chroni przed ucięciem długich rozwiązań matematycznych
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

##################AGENT RAG
from typing import List

class RAGEvaluation(BaseModel):
    """
    Use this schema to return the evaluation of the retrieved examples.
    """
    useful_example_numbers: List[int] = Field(
        description="A list of numbers of the useful examples (e.g., [1, 2]). Return an empty list [] if none are logically useful."
    )


def rag_agent(state: State):
    print("\n--- RAG AGENT: Szukam i oceniam podobne zadania ---")
    original_problem = state["messages"][0].content

    # Wykorzystanie retrievera do pobrania 3 dokumentów
    retrieved_docs = retriever.retrieve(original_problem)

    # Przygotowanie poleceń zadań
    examples_for_evaluation = ""
    for i, doc in enumerate(retrieved_docs, 1):
        examples_for_evaluation += f"--- Example {i} ---\n"
        examples_for_evaluation += f"Problem: {doc.text}\n\n"

    print("zadania wybrane przez agenta RAG:")
    print(examples_for_evaluation)

    # Pobieramy prompt
    prompt = prompts_used.get_rag_eval_prompt(
        original_problem = original_problem,
        rag_found = examples_for_evaluation
    )

    structured_llm = llm_RAG.with_structured_output(RAGEvaluation)
    result = structured_llm.invoke([HumanMessage(content=prompt)])

    # Wyciągamy bezpiecznie listę wygenerowaną przez LLM
    selected_nums = result.useful_example_numbers
    print("Logicznie dopasowane zadania nr:")
    print(selected_nums)

    # 3. BUDOWA KONTEKSTU DLA SOLVERA
    context_to_inject = ""

    if not selected_nums:  # Jeśli lista jest pusta
        print("RAG AGENT: Brak logicznego dopasowania (Zwrócono pustą listę). Odrzucam przykłady.")
    else:
        print(f"RAG AGENT: Sukces! Wybrano logicznie przydatne przykłady: {selected_nums}.")

        for num in selected_nums:
            if 1 <= num <= len(retrieved_docs):
                selected_idx = num - 1
                selected_doc = retrieved_docs[selected_idx]

                solution = selected_doc.metadata.get("solution", "Brak rozwiązania")
                lvl = selected_doc.metadata.get("level", "Unknown")
                math_type = selected_doc.metadata.get("type", "Unknown")

                context_to_inject += (
                    f"--- USEFUL EXAMPLE ---\n"
                    f"Problem: {selected_doc.text}\n"
                    f"Solution: {solution}\n\n"
                )
            else:
                print(f"RAG AGENT: Ostrzeżenie! LLM podał numer spoza zakresu: {num}")

    return {"rag_context": context_to_inject}

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import RemoveMessage

def solver(state: State):
    messages = state["messages"]
    rag_context = state.get("rag_context")
    system_prompt_text = prompts_used.get_solver_prompt(rag_context=rag_context)
    feedback = state.get("feedback")
    if feedback:
        system_prompt_text = system_prompt_text + feedback
    system_prompt = SystemMessage(content= system_prompt_text)
    prompt_with_history = [system_prompt] + messages
    print("Co otrzymuje solver:")
    pprint(prompt_with_history)
    print("koniec wiadomości dla solvera")
    response = llm_with_tools.invoke(prompt_with_history)
    print("Odpowiedź solvera:")
    print(response)
    return {"messages": [response]}

def solver_router(state: State):
    messages = state["messages"]
    #print(f"ilosc wywołań weryfikatora(solver router): {state.get('calls', 0)}")
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "submit_to_verifier":
            print("Router: Solver przekazuje odpowiedź do Weryfikatora.")
            return "verifier"

        print(f"Router: Agent wykonuje obliczenia/szuka w {tool_name}...")
        return "tools"
    return "verifier"


# def old_verifier(state: State):
#     print("--- WERYFIKATOR ---")
#     messages = state["messages"]
#     current_calls = state.get("calls", 0)
#     print(f"Ilość dotychczasowych wywołań weryfikatora: {current_calls}")
#
#     # 1. SZUKAMY CZY SOLVER UŻYŁ NARZĘDZIA DO ZGŁOSZENIA
#     last_msg = messages[-1]
#     tool_call_id = None
#     if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
#         for tc in last_msg.tool_calls:
#             if tc["name"] == "submit_to_verifier":
#                 tool_call_id = tc["id"]
#                 final_ans = tc["args"].get("final_answer_in_latex", "BRAK")
#                 break
#
#     if current_calls >= 3:
#         print("Weryfikator: Osiągnięto limit prób. Wymuszam akceptację.")
#         force_msg = f"Osiągnięto limit weryfikacji. Ostateczny wynik: \\boxed{{{final_ans if tool_call_id else 'ERROR'}}}"
#
#         #jeśli solver czeka na wynik z ToolCall, wynik musi być typu ToolMessage
#         if tool_call_id:
#             return {"messages": [ToolMessage(content=force_msg, tool_call_id=tool_call_id)], "calls": current_calls + 1}
#         else:
#             return {"messages": [AIMessage(content=force_msg)], "calls": current_calls + 1}
#
#     conversation_transcript = ""
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             conversation_transcript += f"USER QUESTION: {msg.content}\n\n"
#         elif isinstance(msg, AIMessage):
#             if msg.tool_calls:
#                 conversation_transcript += f"SOLVER (CODE ATTEMPT):\n{msg.tool_calls}\n"
#             else:
#                 conversation_transcript += f"SOLVER: {msg.content}\n\n"
#         elif isinstance(msg, ToolMessage):
#             conversation_transcript += f"SYSTEM (OUTPUT): {msg.content}\n\n"
#
#     verifier_prompt = prompts_used.get_verifier_prompt(conversation_transcript)
#     print(verifier_prompt)
#     response = llm.invoke([HumanMessage(content=verifier_prompt)])
#
#     if not response.content:
#         feedback_content = "VERDICT: FEEDBACK: Empty response from Verifier. Please double check your calculations and submit again."
#     else:
#         feedback_content = response.content
#
#     if tool_call_id:
#         return {
#             "messages": [ToolMessage(content=f"MESSAGE FROM VERIFIER:\n{feedback_content}", tool_call_id=tool_call_id)],
#             "calls": current_calls + 1
#         }
#     else:
#         return {
#             "messages": [HumanMessage(content=f"MESSAGE FROM VERIFIER:\n{feedback_content}")],
#             "calls": current_calls + 1
#         }

from pydantic import BaseModel, Field

class NewVerificationResult(BaseModel):
    is_correct: bool = Field(
        description="True ONLY if the solution perfectly meets all 5 criteria. False if there is ANY flaw, hallucination, or unhandled tool error."
    )
    answer: str = Field(
        description="If is_correct is True: provide the exact final answer from the solution in \\boxed{final_answer} format. If is_correct is False: return an empty string ''."
    )
    feedback: str = Field(
        description="If is_correct is False: State the exact fail reason and provide a clear instruction for Agent Solver on what to fix. If is_correct is True: return an empty string ''."
    )

def verifier(state: State):
    print("--- WERYFIKATOR ---")
    messages = state["messages"]
    verifier_prompt = prompts_used.new_get_verifier_prompt(create_conversation(messages))
    print(f"\nCO DOSTAJE WERYFIKATOR:{verifier_prompt}\nKONIEC WIADOMOŚCI DO WERYFIKATORA\n")
    structured_llm = llm.with_structured_output(NewVerificationResult)
    response = structured_llm.invoke([HumanMessage(content=verifier_prompt)])
    print("co zwraca weryfikator:\n")
    print(response)
    new_iteration = state.get("iterations", 0) + 1
    to_end_flag = response.is_correct
    answer = []
    #delete_instructions = []
    if new_iteration >= 3:
        to_end_flag = True
    if response.is_correct is True:
        answer = response.answer
    else:
        answer = response.feedback
    # if not to_end_flag:
    #     delete_instructions = [RemoveMessage(id=m.id) for m in state["messages"]]
    return {"messages": answer, "to_end": to_end_flag, "iterations": new_iteration}

def verifier_router(state: State):
    if state["to_end"] is True:
        return END
    return "solver"

# def NEEEW_verifier(state: State):
#     print("--- WERYFIKATOR ---")
#     messages = state["messages"]
#     current_calls = state.get("calls", 0)
#     print(f"Ilość dotychczasowych wywołań weryfikatora: {current_calls}")
#
#     if current_calls >= 3:
#         print("Weryfikator: Osiągnięto limit prób. Wymuszam akceptację.")
#         force_msg = f"Osiągnięto limit weryfikacji - zwracam nieweryfikowany wynik"
#         return {"calls": current_calls + 1}
#
#     conversation_transcript = ""
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             conversation_transcript += f"USER QUESTION: {msg.content}\n\n"
#         elif isinstance(msg, AIMessage):
#             if msg.tool_calls:
#                 conversation_transcript += f"SOLVER (CODE ATTEMPT):\n{msg.tool_calls}\n"
#             else:
#                 conversation_transcript += f"SOLVER: {msg.content}\n\n"
#         elif isinstance(msg, ToolMessage):
#             conversation_transcript += f"SYSTEM (OUTPUT): {msg.content}\n\n"
#
#     verifier_prompt = prompts_used.get_verifier_prompt(conversation_transcript)
#     print(verifier_prompt)
#     response = llm.invoke([HumanMessage(content=verifier_prompt)])
#
#     if not response.content:
#         feedback_content = "VERDICT: FEEDBACK: Empty response from Verifier. Please double check your calculations and submit again."
#     else:
#         feedback_content = response.content
#
#     if tool_call_id:
#         return {
#             "messages": [ToolMessage(content=f"MESSAGE FROM VERIFIER:\n{feedback_content}", tool_call_id=tool_call_id)],
#             "calls": current_calls + 1
#         }
#     else:
#         return {
#             "messages": [HumanMessage(content=f"MESSAGE FROM VERIFIER:\n{feedback_content}")],
#             "calls": current_calls + 1
#         }


# def old_verifier_router(state: State):
#     messages = state["messages"]
#     last_message = messages[-1].content
#     current_calls = state.get("calls", 0)
#
#     print(f"Router Weryfikatora. Aktualne calls: {current_calls}")
#
#     # Jeśli narzuciliśmy akceptację przez limit prób:
#     if "Osiągnięto limit weryfikacji" in last_message or current_calls >= 3:
#         print("Koniec (limit)")
#         return END
#
#     if "VERDICT: FEEDBACK:" in last_message.upper():
#         print("Wracamy do Solvera - błędy znalezione")
#         return "solver"
#
#     if "\\boxed{" in last_message:
#         print("Koniec - wynik zaakceptowany")
#         return END
#
#     print("Fallback - niezrozumiała odpowiedź Weryfikatora. Powrót do Solvera.")
#     return "solver"


#Graf
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

#Zapis wizualizacji grafu
try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("multi_agent_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'multi_agent_graf.png'")
except Exception:
    print("Nie udało się wygenerować grafu")


if __name__ == "__main__":
    #asyncio.run(main())
    evaluate_llm("/home/olacz/Downloads/MATH/test/algebra", "Level 4")
    #evaluate_llm("/home/olacz/Downloads/MATH/test/geometry", "Level 5")
    #evaluate_llm("/home/olacz/Downloads/MATH/test/number_theory", "Level 3")
