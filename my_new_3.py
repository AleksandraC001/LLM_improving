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

#SEKCJA 1: PROSTE WYWOŁANIE LLM
llm_simple = ChatGroq(model_name=MODEL, temperature=0.7, api_key=GROQ_API_KEY)
response = llm_simple.invoke("opowiedz żart")
print(response.content)


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
        To solve the problem you need to use python tool.
        When you receive the message from the tool you need to take the final 
        answer and return it in format: \\boxed{final_answer}
        """)

    if isinstance(messages[0], SystemMessage):
        messages[0] = system_prompt
    else:
        messages = [system_prompt] + messages

    return {"messages": [llm_with_tools.invoke(messages)]}

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
user_input_0 = "ile to 2 + 8?"
user_input = (
    "Oblicz sumę wszystkich liczb pierwszych w zakresie od 1 do 500, "
    "a następnie sprawdź, czy ta suma jest podzielna przez 7. "
    "Podaj wynik sumy oraz wynik dzielenia z resztą."
)
user_input1 = (
    "Oblicz 100-tny wyraz ciągu Fibonacciego. "
    "Następnie oblicz sumę cyfr tej ogromnej liczby i podaj wynik."
)
user_input2 = (
    "Przeprowadź symulację Monte Carlo, aby przybliżyć wartość liczby Pi. "
    "Wygeneruj 10,000 losowych punktów w kwadracie 1x1 i sprawdź, "
    "ile z nich wpada w koło o promieniu 1. Podaj uzyskany wynik przybliżenia."
)
user_input3 = (
    "Oblicz dokładną liczbę dni, jaka minęła od daty urodzenia Alana Turinga "
    "(23 czerwca 1912) do dzisiaj. Następnie sprawdź, jaki to był dzień "
    "tygodnia w dniu jego urodzin."
)

state = graph.invoke({"messages": [{"role": "user", "content": user_input_0}]})
print(state["messages"][-1].content)
print('\n')
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
for msg in state["messages"]:
    pprint(msg)
print(state["messages"][-1].content)
print('\n')
state = graph.invoke({"messages": [{"role": "user", "content": user_input1}]})
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
print(state["messages"][-1].content)

# Wizualizacja grafu
try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("moj_agent_graf_new.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'moj_agent_graf_new.png'")
except Exception:
    print("Nie udało się wygenerować grafu")