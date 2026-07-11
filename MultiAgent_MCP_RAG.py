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

# Odczytanie bazy zadań:
path = '/home/olacz/Downloads/MATH/train/'
topics = os.listdir(path)
print("Files and directories in '", path, "' :")
print(topics)

#ładuję dokumentu za zbioru math - w metadanych dodaję stopień trudności, rozwiązanie i typ zadania
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
# załadowaie tematów:
path = '/home/olacz/Downloads/MATH/train/'
topics = os.listdir(path)
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
    RIGHT_RAW = 2
    RIGHT_AFTER_CLEANING = 3
    RIGHT_LATEX = 4


def extract_last_boxed_answer(text: str) -> str:
    """
    Wyciąga zawartość ostatniego tagu \boxed{}, poprawnie
    obsługując zagnieżdżone nawiasy klamrowe.
    """
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return ""

    start = idx + 7
    open_brackets = 1

    for i in range(start, len(text)):
        if text[i] == '{':
            open_brackets += 1
        elif text[i] == '}':
            open_brackets -= 1

        if open_brackets == 0:
            return text[start:i]
    return ""


def evaluate_model_on_problem(data) -> ModelEvaluation:
    # 1. Wyciągnięcie poprawnej odpowiedzi z klucza "solution"
    # Szukamy zawartości \boxed{...}
    match_right = re.search(r'\\boxed{(.*)}', data["solution"])
    right_answer = match_right.group(1) if match_right else ""

    if not right_answer:
        print(f"BŁĄD: Nie znaleziono poprawnej odpowiedzi w formacie \\boxed{{}} w kluczu solution.")

    # preprocessing poprawnej, oczekiwanej odpowiedzi:
    right_answer = preprocessing_2.clean_math_answer(right_answer)

    # 2. Przygotowanie wiadomości dla agenta
    inputs = {"messages": [HumanMessage(content=data["problem"])]}
    try:
        # Wywołujemy skompilowany graf
        # result = graph.invoke(inputs)
        result = graph.invoke({"messages": [{"role": "user", "content": data["problem"]}]},
                              config={"recursion_limit": 12})
        for i in result["messages"]:
            pprint(i)
        print(result["messages"][-1].content)
        print('\n')
    except httpx.ReadTimeout:
        print("Timeout")
    except openai.APITimeoutError:  # <-- DODAJ TEN WYJĄTEK
        print("Timeout - Endpoint API nie odpowiedział w wyznaczonym czasie")
        return ModelEvaluation.ERROR
    except GraphRecursionError:
        print("Recursion limit exceeded (Model się zapętlił)")
        return ModelEvaluation.WRONG  # Traktujemy pętlę jako błąd rozwiązania
    except openai.BadRequestError as e:
        print(f"Bad request - SZCZEGÓŁY BŁĘDU: {e}")
        return ModelEvaluation.ERROR
    except Exception as e:
        print(f"Inny błąd: {e}")
        return ModelEvaluation.ERROR

    # 3. Pobranie ostatniej wiadomości i wyciągnięcie odpowiedzi z \boxed{...}
    last_message = result['messages'][-1].content
    print(f"OSTATNIA WIADOMOSC: {last_message}")

    # matches_llm = re.findall(r'\\boxed{(.*?)}', last_message)
    #
    # if not matches_llm:
    #     print(f"Nie znalazł boxed w odpowiedzi modelu.")
    #     return ModelEvaluation.WRONG
    #
    # # Pobieramy ostatnią znalezioną ramkę, która niemal zawsze jest ostatecznym wynikiem
    # llm_answer = matches_llm[-1]
    llm_answer = extract_last_boxed_answer(last_message)

    if not llm_answer:
        print(f"Nie znalazł boxed w odpowiedzi modelu.")
        return ModelEvaluation.WRONG
    llm_answer = preprocessing_2.clean_math_answer(llm_answer)

    try:
        if sympify(right_answer).equals(sympify(llm_answer)):
            return ModelEvaluation.RIGHT_RAW
        # Jeśli sympify zadziałało, ale wynik jest inny
        else:
            print(f"BŁĄD MATEMATYCZNY (smf): powinno być: {right_answer}, a jest: {llm_answer}")
            return ModelEvaluation.WRONG
    except Exception as e:
        if right_answer == llm_answer:
            return ModelEvaluation.RIGHT_RAW
        else:
            print(f"powinno być: {right_answer}, a jest: {llm_answer}")
            # Możesz tu dodać dodatkową logikę czyszczenia stringów,
            # jeśli np. spacja lub formatowanie LaTeX robi różnicę
            return ModelEvaluation.WRONG


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
            # print(data)
            print(f"aktualny stan: {i=}, {c=}")
            model_evaluation_on_problem = evaluate_model_on_problem(data)
            c[model_evaluation_on_problem] += 1
            i += 1


from MCP_tools_v1 import tools

# SEKCJA 4: LOGIKA AGENTA (LANGGRAPH)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    calls: Annotated[int, lambda x, y: x + y]
    rag_context: str | None

from langchain_openai import ChatOpenAI
import openai

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

def solver(state: State):
    messages = state["messages"]
    print(f"główny solver działa, dostaje:{state["messages"][-1].content}")

    rag_context = state.get("rag_context")
    system_prompt_text = prompts_used.get_solver_prompt(rag_context=rag_context)
    print("\nPROMPT SYSTEMU DLA SOLVERA:\n")
    print(f"{system_prompt_text}\n")
    system_prompt = SystemMessage(content= system_prompt_text)
    prompt_with_history = [system_prompt] + messages
    response = llm_with_tools.invoke(prompt_with_history)

    return {"messages": [response]}

###################_________________________________AGENT RAG___________________________________
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

    # 1. Wykorzystanie retrievera do pobrania 3 dokumentów
    retrieved_docs = retriever.retrieve(original_problem)

    # Przygotowanie "chudych" przykładów dla LLM (tylko polecenia)
    examples_for_evaluation = ""
    for i, doc in enumerate(retrieved_docs, 1):
        examples_for_evaluation += f"--- Example {i} ---\n"
        examples_for_evaluation += f"Problem: {doc.text}\n\n"

    # Pobieramy prompt
    prompt = prompts_used.get_rag_eval_prompt(
        original_problem = original_problem,
        rag_found = examples_for_evaluation
    )

    structured_llm = llm_RAG.with_structured_output(RAGEvaluation)
    result = structured_llm.invoke([HumanMessage(content=prompt)])

    # Wyciągamy bezpiecznie listę wygenerowaną przez LLM
    selected_nums = result.useful_example_numbers
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

def verifier(state: State):
    print("weryfikator")
    messages = state["messages"]

    print("weryfikator")
    messages = state["messages"]

    # Pobieramy obecną liczbę wywołań
    current_calls = state.get("calls", 0)
    print(f"ilosc wywołań weryfikatora: {current_calls}")

    if current_calls >= 2:  # Zakładając, że zaczynamy od 0, więc 0, 1, 2 to trzy próby
        print("Weryfikator: Osiągnięto limit prób. Wymuszam akceptację wyniku Solvera.")

        # Przeszukujemy historię od końca, żeby znaleźć, co Solver przekazał do narzędzia
        for msg in reversed(messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "submit_to_verifier":
                        # Wyciągamy argument, który podał Solver
                        final_ans = tc["args"].get("final_answer_in_latex", "BRAK")
                        # Zwracamy sztuczną wiadomość jako Weryfikator, zawierającą \boxed{}
                        # Dzięki temu Twój skrypt ewaluacyjny z łatwością to wyłapie!
                        return {"messages": [AIMessage(
                            content=f"Osiągnięto limit weryfikacji. Ostateczny wynik: \\boxed{{{final_ans}}}")],
                                "calls": 1}

        # Fallback w razie dziwnego błędu
        return {"messages": [AIMessage(content="Błąd wyciągania wyniku. \\boxed{ERROR}")], "calls": 1}

    state["calls"] = state.get("calls", 0) + 1

    print(f"ilosc wywołań weryfikatora:{state.get("calls", "puste")}")
    # 1. Budujemy "scenariusz" rozmowy w czystym tekście.
    # To ukrywa przed modelem skomplikowaną strukturę ToolCalli, która go zawiesza.
    conversation_transcript = ""

    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_transcript += f"USER QUESTION: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Wyciągamy kod, który Solver chciał uruchomić
                for tc in msg.tool_calls:
                    args = tc.get('args', '')
                    conversation_transcript += f"SOLVER (CODE ATTEMPT):\n{args}\n"
            else:
                conversation_transcript += f"SOLVER (TEXT RESPONSE): {msg.content}\n\n"
        elif isinstance(msg, ToolMessage):
            conversation_transcript += f"SYSTEM (CODE OUTPUT): {msg.content}\n\n"

    # 2. Tworzymy jeden jasny prompt z wklejonym scenariuszem
    # print("co dostał weryfikator:")
    # print(audit_prompt)
    # print("koniec wiadomości do weryfikatora")

    verifier_prompt = prompts_used.get_verifier_prompt(conversation_transcript)
    response = llm.invoke([HumanMessage(content=verifier_prompt)])

    if not response.content:
        # Przekazujemy to jako HumanMessage po angielsku, aby Solver potraktował to jako komendę systemową
        fallback_msg = "SYSTEM MESSAGE: The Verifier did not provide any feedback, but a final answer is still missing. Please conclude your calculations and use the 'submit_to_verifier' tool to provide the final result."
        return {"messages": [HumanMessage(content=fallback_msg)]}

        # Jeśli model normalnie odpowiedział, przepakowujemy jego odpowiedź:
    feedback_content = response.content
    return {"messages": [HumanMessage(content=f"MESSAGE FROM VERIFIER:\n{feedback_content}")], "calls": 1}    # Zabezpieczenie na wypadek, gdyby model i tak zwrócił pusto (bardzo rzadkie przy tej metodzie)



    # KRYTYCZNA ZMIANA: Przepakowujemy AIMessage z Weryfikatora w HumanMessage,
    # żeby Solver traktował to jako nową instrukcję z zewnątrz.
    return {"messages": [HumanMessage(content=f"Wiadomość od Systemu Weryfikującego:\n{feedback_content}")], "calls": 1}


def solver_router(state: State):
    messages = state["messages"]
    print(f"ilosc wywołań weryfikatora(solver router): {state.get('calls', 0)}")

    last_message = messages[-1]

    # Model dzięki `tool_choice="required"` zawsze zwraca tool_calls, więc wchodzimy tu od razu:
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]

        # 1. Sprawdzamy, czy agent "wcisnął przycisk" weryfikacji
        if tool_name == "submit_to_verifier":
            print("Router: Agent skończył, przekazuję do Weryfikatora.")
            return "verifier"

        # 2. Jeśli użył czegokolwiek innego (Python, Arxiv, Brave), idziemy do narzędzi
        print(f"Router: Agent wykonuje obliczenia/szuka w {tool_name}...")
        return "tools"

    # --- BEZPIECZNIK (Fallback) ---
    # Jeśli wystąpiłby dziwny błąd API i model nie wezwał narzędzia,
    # wymuszamy przejście do węzła "tools", co wygeneruje techniczny błąd,
    # a LangGraph i tak wymusi na modelu poprawę.
    return "verifier"###############################################################################czy na pewno?


def verifier_router(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content
    # state["calls"] = state.get("calls", 0) + 1
    print(f"ilosc wywołań weryfikatora:{state.get("calls", "puste")}")
    if "FEEDBACK:" in content and state["calls"] < 3:
        print("wracamy do solvera")
        return "solver"

    if "\\boxed{" in content or state["calls"] >= 3:
        print("konczymy")
        return END

    return END

# Budowa grafu
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

# Wizualizacja grafu
try:
    print("Generowanie wizualizacji grafu...")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("multi_agent_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graf został zapisany jako 'multi_agent_graf.png'")
except Exception:
    print("Nie udało się wygenerować grafu")


#state = graph.invoke({"messages": [{"role": "user", "content": test_query}]}, config={"recursion_limit": 12})

#print("\n--- OSTATNIA ODPOWIEDŹ AGENTA ---")
#print(state["messages"][-1].content)

#evaluate_llm("/home/olacz/Downloads/MATH/test/number_theory", "Level 3")
#evaluate_llm("/home/olacz/Downloads/MATH/test/algebra", "Level 3")


###########################asynchroniczne testowanie

import asyncio


async def evaluate_model_on_problem_async(data, semaphore, filename) -> ModelEvaluation:
    ocena_koncowa = None
    uwagi_o_bledach = ""
    result = None
    llm_answer = "BRAK"
    right_answer = ""

    # 1. Wyciągnięcie poprawnej odpowiedzi z klucza "solution"
    match_right = re.search(r'\\boxed{(.*)}', data["solution"])
    if match_right:
        right_answer = match_right.group(1)
        right_answer = preprocessing_2.clean_math_answer(right_answer)
    else:
        uwagi_o_bledach = "Nie znaleziono poprawnej odpowiedzi w formacie \\boxed{} w kluczu solution."
        print(f"BŁĄD: {uwagi_o_bledach}")

    # 2. Asynchroniczne wywołanie grafu
    async with semaphore:
        try:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": data["problem"]}]},
                config={"recursion_limit": 12}
            )
            print(f"Zakończono zadanie pomyślnie: {filename}")
        except httpx.ReadTimeout:
            uwagi_o_bledach = "Timeout"
            ocena_koncowa = ModelEvaluation.ERROR
        except GraphRecursionError:
            uwagi_o_bledach = "Recursion limit exceeded (Model się zapętlił)"
            ocena_koncowa = ModelEvaluation.WRONG
        except openai.BadRequestError as e:
            uwagi_o_bledach = f"Bad request - SZCZEGÓŁY BŁĘDU: {e}"
            ocena_koncowa = ModelEvaluation.ERROR
        except Exception as e:
            uwagi_o_bledach = f"Inny błąd grafu: {e}"
            ocena_koncowa = ModelEvaluation.ERROR

    # 3. Pobranie ostatniej wiadomości i sprawdzanie wyniku (Tylko jeśli graf zadziałał)
    if ocena_koncowa is None and result is not None:
        last_message = result['messages'][-1].content
        matches_llm = re.findall(r'\\boxed{(.*?)}', last_message)

        if not matches_llm:
            uwagi_o_bledach = "Nie znalazł boxed w odpowiedzi modelu."
            ocena_koncowa = ModelEvaluation.WRONG
        else:
            llm_answer = matches_llm[-1]
            llm_answer = preprocessing_2.clean_math_answer(llm_answer)

            try:
                if sympify(right_answer).equals(sympify(llm_answer)):
                    ocena_koncowa = ModelEvaluation.RIGHT_RAW
                else:
                    uwagi_o_bledach = f"BŁĄD MATEMATYCZNY (smf): powinno być: {right_answer}, a jest: {llm_answer}"
                    ocena_koncowa = ModelEvaluation.WRONG
            except Exception as e:
                if right_answer == llm_answer:
                    ocena_koncowa = ModelEvaluation.RIGHT_RAW
                else:
                    uwagi_o_bledach = f"BŁĄD PORÓWNANIA: powinno być: {right_answer}, a jest: {llm_answer}"
                    ocena_koncowa = ModelEvaluation.WRONG

    # Zabezpieczenie, jeśli żaden warunek nie przypisał oceny
    if ocena_koncowa is None:
        ocena_koncowa = ModelEvaluation.ERROR

    # ---------------------------------------------------------
    # 4. TWORZENIE CZYTELNEGO RAPORTU Z MYŚLENIA (ŚLAD AGENTA)
    # ---------------------------------------------------------
    raport = f"# PLIK ZADANIA: {filename}\n\n"
    raport += f"### PROBLEM:\n{data['problem']}\n\n"
    raport += f"### OCZEKIWANA ODPOWIEDŹ (Z BAZY):\n**{right_answer}**\n\n"
    raport += f"### ODPOWIEDŹ MODELU (WYCIĄGNIĘTA):\n**{llm_answer}**\n\n"
    raport += "---\n## PRZEBIEG ROZUMOWANIA (LANGGRAPH):\n---\n\n"

    # Wyciągamy krok po kroku wszystko, co działo się w grafie
    if result is not None and "messages" in result:
        for msg in result["messages"]:
            raport += f"### [{msg.__class__.__name__.upper()}]\n"

            # Jeśli agent użył narzędzia, wypiszmy jakiego i z jakimi danymi
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool in msg.tool_calls:
                    raport += f"> **WYWOŁUJE NARZĘDZIE:** `{tool['name']}`\n"
                    raport += f"> **ARGUMENTY:**\n```json\n{tool['args']}\n```\n"

            # Wypiszmy tekst wiadomości
            if msg.content:
                raport += f"{msg.content}\n"

            raport += "\n---\n"
    else:
        raport += "*Graf nie zwrócił historii wiadomości (prawdopodobnie wystąpił błąd krytyczny).*\n\n"

    if uwagi_o_bledach:
        raport += f"\n### UWAGI / BŁĘDY:\n{uwagi_o_bledach}\n"

    raport += f"\n## WYNIK EWALUACJI: {ocena_koncowa.name}\n"

    # Zapisujemy do folderu 'raporty_agentow'
    os.makedirs("raporty_agentow", exist_ok=True)
    with open(f"raporty_agentow/{filename}.md", "w", encoding="utf-8") as f:
        f.write(raport)

    # Ostateczny zwrot dla Countera
    return ocena_koncowa


async def evaluate_llm_async(dir_, lvl, max_tasks=20):
    print(f"\n--- Rozpoczynam asynchroniczną ewaluację {dir_} na poziomie {lvl} ---")

    semaphore = asyncio.Semaphore(5)
    tasks = []

    i = 0
    for filename in os.listdir(dir_):
        if i >= max_tasks:
            break

        with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
            if data["level"] == lvl:
                # TUTAJ JEST KLUCZOWA ZMIANA:
                # Przekazujemy do Solvera 3 rzeczy: dane zadania, semafor i właśnie NAZWĘ PLIKU
                tasks.append(evaluate_model_on_problem_async(data, semaphore, filename))
                i += 1

    print(f"Kolejkuję {len(tasks)} zadań do serwera vLLM...")

    # Odpalenie wszystkich zadań równolegle (tu dzieje się cała asynchroniczna magia)
    results = await asyncio.gather(*tasks)

    # Zliczanie wyników
    c = Counter(results)
    print(f"\nFINAŁ dla {dir_}: {c}")
    return c

async def main():
    # Odpalamy najpierw jedną paczkę zadań...
#    await evaluate_llm_async("/home/olacz/Downloads/MATH/test/number_theory", "Level 5", max_tasks=20)
    # ...a potem drugą (możesz usunąć limit max_tasks, jeśli chcesz przetestować wszystko)
    await evaluate_llm_async("/home/olacz/Downloads/MATH/test/algebra", "Level 4", max_tasks=10)
    #await evaluate_llm_async("/home/olacz/Downloads/MATH/test/geometry", "Level 4", max_tasks=20)


if __name__ == "__main__":
    asyncio.run(main())
    #evaluate_llm("/home/olacz/Downloads/MATH/test/algebra", "Level 4")
    #evaluate_llm("/home/olacz/Downloads/MATH/test/geometry", "Level 5")
