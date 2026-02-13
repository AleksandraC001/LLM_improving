import os
import json
from pprint import pprint
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from groq import Groq
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun

from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

#Odczytanie bazy zadań:
path = '/home/olacz/Downloads/MATH/train/'
topics = os.listdir(path)
print("Files and directories in '", path, "' :")
print(topics)

#KONFIGURACJA I KLUCZE
with open("groq_api_key", "r") as f:
    GROQ_API_KEY = f.read().strip()

MODEL = 'llama-3.3-70b-versatile'

client = Groq(api_key='api_key') # Zostawiam 'api_key' zgodnie z oryginałem

'''#SEKCJA 1: PROSTE WYWOŁANIE LLM
llm_simple = ChatGroq(model_name=MODEL, temperature=0.7, api_key=GROQ_API_KEY)
response = llm_simple.invoke("opowiedz żart")
print(response.content)'''

import os
import json
from llama_index.core.schema import Document


def load_math_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            content = f"Problem:\n{data['problem']}\n\nSolution:\n{data['solution']}"
            doc = Document(text=content)
            documents.append(doc)
    return documents

documents = []
for topic in topics:
    documents.extend(load_math_documents(os.path.join(path, topic)))
from collections import Counter

import re
import httpx
import groq
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from enum import Enum

class ModelEvaluation(Enum):
    ERROR = 0
    WRONG = 1
    RIGHT_RAW = 2
    RIGHT_AFTER_CLEANING = 3
    RIGHT_LATEX = 4

def evaluate_model_on_problem(data) -> ModelEvaluation:
    # 1. Wyciągnięcie poprawnej odpowiedzi z klucza "solution"
    # Szukamy zawartości \boxed{...}
    match_right = re.search(r'\\boxed{(.*)}', data["solution"])
    right_answer = match_right.group(1) if match_right else ""

    if not right_answer:
        print(f"BŁĄD: Nie znaleziono poprawnej odpowiedzi w formacie \\boxed{{}} w kluczu solution.")

    # 2. Przygotowanie wiadomości dla agenta
    inputs = {"messages": [HumanMessage(content=data["problem"])]}
    try:
        # Wywołujemy skompilowany graf
        #result = graph.invoke(inputs)
        result = graph.invoke({"messages": [{"role": "user", "content": data["problem"]}]})
        for i in result["messages"]:
            pprint(i)
        print(result["messages"][-1].content)
        print('\n')
    except httpx.ReadTimeout:
        print("Timeout")
        return ModelEvaluation.ERROR
    except groq.BadRequestError:
        print("Bad request")
        return ModelEvaluation.ERROR
    except GraphRecursionError:
        print("Recursion limit exceeded")
        return ModelEvaluation.ERROR
    except Exception as e:
        print(f"Inny błąd: {e}")
        return ModelEvaluation.ERROR

    # 3. Pobranie ostatniej wiadomości i wyciągnięcie odpowiedzi z \boxed{...}
    last_message = result['messages'][-1].content
    print(f"OSTATNIA WIADOMOSC: {last_message}")

    match_llm = re.search(r'\\boxed{(.*)}', last_message)

    if not match_llm:
        print(f"Nie znalazł boxed w odpowiedzi modelu.")
        return ModelEvaluation.WRONG

    llm_answer = match_llm.group(1)

    # 4. Porównanie odpowiedzi
    if right_answer == llm_answer:
        return ModelEvaluation.RIGHT_RAW
    else:
        # Możesz tu dodać dodatkową logikę czyszczenia stringów,
        # jeśli np. spacja lub formatowanie LaTeX robi różnicę
        return ModelEvaluation.WRONG

def evaluate_llm(dir_, lvl):
    print(f"Evaluating {dir_} on {lvl}")
    i = 0
    c = Counter()
    for filename in os.listdir(dir_):
        with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)  # Parsowanie JSON do obiektu Pythona (słownik lub lista)
            if data["level"] != lvl:
                continue
            # print(data)
            print(f"aktualny stan: {i=}, {c=}")
            i += 1
            model_evaluation_on_problem = evaluate_model_on_problem(data)
            c[model_evaluation_on_problem] += 1
            if i > 200:
                break

#SEKCJA 3: DEFINICJA NARZĘDZI AGENTA
python_tool = PythonREPLTool()

@tool
def python_interpreter(code: str):
    """
    Uruchamia kod Pythona i zwraca wynik (stdout).
    Używaj do wykonywania potrzebnych obliczeń.
    """
    return python_tool.run(code)

tools = [python_interpreter]

#SEKCJA 4: LOGIKA AGENTA (LANGGRAPH)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0,
    api_key=GROQ_API_KEY
)
llm_with_tools = llm.bind_tools(tools)

def agent(state: State):
    messages = state["messages"]
    system_prompt = SystemMessage(content="""
        You are a helpful mathematical assistant with access to tools.
        Solve the problem step-by-step. U need to use python to solve the problem. Do not solve it by yourself.
        If the problem is complex don't solve it all at once. Instead:
        - Use Python to solve only the first logical sub-problem.
        - Look at the observation.
        - Then use Python again for the next sub-problem.
        - If the answer from python tool is empty call the previous step with the python tool again. 
         
        OUTPUT FORMAT:
        - When you reach the final answer from the tool you must return it in LaTeX box: \\boxed{answer}
        - Example: \\boxed{42}
        
        """)

    prompt_with_history = [system_prompt] + messages
    response = llm_with_tools.invoke(prompt_with_history)
    return {"messages": [response]}

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Budowa grafu
tool_node = ToolNode(tools)
graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()

#URUCHOMIENIE I TESTY
from zadania_testowe import *

state = graph.invoke({"messages": [{"role": "user", "content": user_input_0}]})
print(state["messages"][-1].content)
print('\n')
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
for msg in state["messages"]:
    pprint(msg)
print(state["messages"][-1].content)
print('\n')
'''state = graph.invoke({"messages": [{"role": "user", "content": user_input1}]})
for msg in state["messages"]:
    pprint(msg)
print(state["messages"][-1].content)
print('\n')
state = graph.invoke({"messages": [{"role": "user", "content": user_input2}]})
for msg in state["messages"]:
    pprint(msg)
print(state["messages"][-1].content)
print('\n')
state = graph.invoke({"messages": [{"role": "user", "content": user_input3}]})
for msg in state["messages"]:
    pprint(msg)
print(state["messages"][-1].content)'''

# Wizualizacja grafu
try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("moj_agent_graf_new.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'moj_agent_graf_new.png'")
except Exception:
    print("Nie udało się wygenerować grafu")

evaluate_llm("/home/olacz/Downloads/MATH/test/number_theory", "Level 5")