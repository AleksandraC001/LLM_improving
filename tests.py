import asyncio
import csv
import hashlib
import json
import os
import re
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime
from enum import Enum

import httpx
import openai
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langsmith import Client

from graph_builders.graph_builder import Graph
from prompts_used import llm_as_a_judge_prompt

ls_client = Client()
import langsmith as ls

from dataset import Dataset

MAX_CONCURRENT_TASKS = 80


class ModelEvaluation(Enum):
    ERROR = 0
    WRONG = 1
    RIGHT = 2


evaluator_llm = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key=None,
    base_url="http://localhost:8001/v1",
    temperature=0,
    max_tokens=2000,
    timeout=400.0,
    max_retries=0
)


def make_task_id(task_content: str, length: int = 16) -> str:
    encoded_content = task_content.encode('utf-8')
    hash_object = hashlib.sha256(encoded_content)
    full_hash = hash_object.hexdigest()
    return full_hash[:length]


class Evaluator:
    def save_report(self, report_name, category_catalog, problem, solution, conversation, llm_decision, evaluation,
                    responce_time,
                    in_tokens, out_tokens, tool_use, python_use, wolfram_use, brave_use, rag_use):
        report_data = {
            "task_file": f"{report_name}",
            "zadanie": problem,
            "oczekiwana_odpowiedz": solution,
            "pelny_przebieg_rozumowania": conversation,
            "decyzja_sedziego": llm_decision,
            "wynik_ewaluacji": evaluation,
            "czas_wykonania": responce_time,
            "zuzycie_tokenow": {
                "wejsciowe": in_tokens,
                "wyjsciowe": out_tokens,
                "suma": in_tokens + out_tokens},
            "uzycie_narzedzi": tool_use,
            "uzycie_python": python_use,
            "uzycie_wolfram": wolfram_use,
            "uzycie_brave": brave_use,
            "uzycie_rag": rag_use
        }
        file_path = os.path.join(category_catalog, f"{report_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        # print("Zapisano raport przebiegu zadań")

    def save_report_to_csv(self, report_name, problem, solution, conversation, llm_decision, evaluation, response_time,
                           in_tokens, out_tokens, tool_use, python_use, wolfram_use, brave_use, rag_use):
        file_path = os.path.join(self.results_path, "raport_ewaluacji.csv")
        file_exists = os.path.isfile(file_path)
        report_data = {
            "task_file": report_name,
            "zadanie": problem,
            "oczekiwana_odpowiedz": solution,
            "pelny_przebieg_rozumowania": conversation,
            "decyzja_sedziego": llm_decision,
            "wynik_ewaluacji": evaluation,
            "czas_wykonania": response_time,
            "tokeny_wejsciowe": in_tokens,
            "tokeny_wyjsciowe": out_tokens,
            "tokeny_suma": in_tokens + out_tokens,
            "uzycie_narzedzi": tool_use,
            "uzycie_python": python_use,
            "uzycie_wolfram": wolfram_use,
            "uzycie_brave": brave_use,
            "uzycie_rag": rag_use
        }

        with open(file_path, "a", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=report_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(report_data)

    async def evaluate_model_on_problem(self, data, category_catalog, semaphore: asyncio.Semaphore,
                                        stats_counter: Counter,
                                        verifier: bool) -> ModelEvaluation:
        async with semaphore:
            problem = re.sub(r'\[asy].*?\[/asy]', '', data["problem"], flags=re.DOTALL)
            run_id = hashlib.sha256(problem.encode('utf-8')).hexdigest()
            main_run_id = str(uuid.uuid4())
            report = f"raport_zadanie_{run_id[:8]}"
            execution_time = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            tool_use = 0
            python_use = 0
            brave_use = 0
            wolfram_use = 0
            rag_use = 0
            error = None
            try:
                logs = []
                conversation = []
                logs.append(
                    f"\n\n___________________________________\nNowe zadanie: {data['problem']}\n____________________________________\n")
                start_time = time.perf_counter()
                try:
                    result = await self.graph.graph.ainvoke({"messages": [{"role": "user", "content": problem}]},
                                                            config={"recursion_limit": 25, "run_id": main_run_id})
                finally:
                    end_time = time.perf_counter()
                    execution_time = round(end_time - start_time, 2)
                for msg in result["messages"]:
                    if msg.type == "ai":
                        usage = getattr(msg, "usage_metadata", {}) or {}
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                    if msg.type == "tool":
                        tool_use += 1
                        tool_name = msg.name.lower()
                        if "python_interpreter" in tool_name:
                            python_use += 1
                        elif "brave_search" in tool_name:
                            brave_use += 1
                        elif "ask_wolfram" in tool_name:
                            wolfram_use += 1
                conversation.append(
                    f"[{result["system_prompt_text"].type.upper()}]:\n{result["system_prompt_text"].content}\n")
                for msg in result["messages"]:
                    role = msg.type.upper()
                    logs.append(f"[{role}]:\n{msg.content}\n")
                    conversation.append(f"[{role}]:\n{msg.content}\n")
                    tool_calls = getattr(msg, "tool_calls", []) or []
                    if tool_calls:
                        logs.append(f"[AI_TOOL_CALLS]:\n{tool_calls}\n")
                        conversation.append(f"[AI_TOOL_CALLS]:\n{tool_calls}\n")
                if "HELPFUL CONTEXT FROM RAG" in result["system_prompt_text"].content:
                    rag_use = 1
            except httpx.ReadTimeout:
                error = "Timeout"
            except openai.APITimeoutError:
                error = "Timeout - Endpoint API nie odpowiedział w wyznaczonym czasie"
            except GraphRecursionError:
                error = "Recursion limit exceeded (Model się zapętlił)"
            except openai.BadRequestError as e:
                error = f"Bad request - SZCZEGÓŁY BŁĘDU: {e}"
            except Exception as e:
                error = f"Inny błąd: {e}"
            if error:
                evaluation = ModelEvaluation.ERROR
                self.save_report(f"{report}", category_catalog, problem, error, conversation, "EMPTY",
                                 evaluation.name, execution_time, total_input_tokens, total_output_tokens, tool_use,
                                 python_use,
                                 wolfram_use, brave_use, rag_use)
                self.save_report_to_csv(f"{report}", problem, error, conversation, "EMPTY",
                                        evaluation.name, execution_time, total_input_tokens, total_output_tokens,
                                        tool_use,
                                        python_use, wolfram_use, brave_use, rag_use)
                ls_client.create_feedback(
                    run_id=main_run_id,
                    key="accuracy",
                    score=0.0
                )
                stats_counter[evaluation] += 1
            else:
                last_messages = result["messages"][-1]

                if verifier:
                    last_messages = result["messages"][-3:-1]

                eval_prompt = llm_as_a_judge_prompt(data["solution"], last_messages)
                # with tracing_v2_enabled(project_name="Ewaluacja_LLM_Judge"):
                with ls.tracing_context(enabled=False):
                    try:
                        eval_response = await evaluator_llm.ainvoke([HumanMessage(content=eval_prompt)])
                        llm_decision = eval_response.content.strip().upper()
                    except Exception as eval_e:
                        llm_decision = f"BŁĄD SĘDZIEGO: {eval_e}"
                evaluation = ModelEvaluation.ERROR
                logs.append(f"--- EWALUACJA ---")
                logs.append(f"Oczekiwana: {data['solution']}")
                if "YES" in llm_decision:
                    logs.append(
                        f"odpowiedź systemu: {last_messages}, \n odpowiedź oczekiwana: {data['solution']}\n Odpowiedzi zgodne - uzasadnienie: {llm_decision}\n\n")
                    evaluation = ModelEvaluation.RIGHT
                elif "NO" in llm_decision:
                    logs.append(
                        f"odpowiedź systemu: {last_messages}, \n odpowiedź oczekiwana: {data['solution']}\n Odpowiedzi niezgodne - uzasadnienie: {llm_decision}\n\n")
                    evaluation = ModelEvaluation.WRONG

                if main_run_id and evaluation != ModelEvaluation.ERROR:
                    score = 1.0 if evaluation == ModelEvaluation.RIGHT else 0.0
                    ls_client.create_feedback(
                        run_id=main_run_id,
                        key="accuracy",
                        score=score
                    )

                stats_counter[evaluation] += 1
                logs.append(f"Aktualne statystyki: {', '.join(f'{k.name}: {v}' for k, v in stats_counter.items())}")
                # print("\n".join(logs))
                self.save_report(f"{report}", category_catalog, problem, data.get("solution", "Brak"), conversation,
                                 llm_decision,
                                 evaluation.name, execution_time, total_input_tokens, total_output_tokens, tool_use,
                                 python_use,
                                 wolfram_use, brave_use, rag_use)
                self.save_report_to_csv(f"{report}", problem, data.get("solution", "Brak"), conversation, llm_decision,
                                        evaluation.name, execution_time, total_input_tokens, total_output_tokens,
                                        tool_use,
                                        python_use, wolfram_use, brave_use, rag_use)
            return evaluation

    async def evaluate_llm(self, dir_, verifier):
        category_catalog = os.path.join(self.results_path, dir_)
        os.makedirs(category_catalog, exist_ok=True)
        print(f"Evaluating {dir_}")
        logs = [f"Evaluating {dir_}"]
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        stats_counter = Counter()
        tasks = []
        try:
            filenames = sorted(os.listdir(dir_))
        except FileNotFoundError:
            print(f"Brak katalogu {dir_}")
            return

        for filename in filenames:
            if len(tasks) >= 2:
                break
            with open(os.path.join(dir_, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                task = self.evaluate_model_on_problem(data, category_catalog, semaphore, stats_counter, verifier)
                tasks.append(task)

        print(f"Znaleziono {len(tasks)} zadań do przetworzenia.")
        logs.append(f"Znaleziono {len(tasks)} zadań do przetworzenia.")

        await asyncio.gather(*tasks)
        print("\n--- Zakończono ewaluację ---")
        print(f"Końcowe statystyki: {', '.join(f'{k.name}: {v}' for k, v in stats_counter.items())}")
        logs.append("\n--- Zakończono ewaluację ---")
        logs.append(f"Końcowe statystyki: {', '.join(f'{k.name}: {v}' for k, v in stats_counter.items())}")
        stats_dict = {k.name: v for k, v in stats_counter.items()}
        right = stats_dict.get("RIGHT", 0)
        wrong = stats_dict.get("WRONG", 0)
        error = stats_dict.get("ERROR", 0)
        print("\n--- PODSUMOWANIE WYDAJNOŚCI ---")
        logs.append("\n--- PODSUMOWANIE WYDAJNOŚCI ---")
        time_list = []
        tokens_in_list = []
        tokens_out_list = []
        tokens_all_list = []
        python_use = 0
        brave_use = 0
        wolfram_use = 0
        rag_use = 0
        tool_use = 0
        tasks_with_tool_use = 0
        for filename in os.listdir(category_catalog):
            if filename.endswith(".json"):
                file_path = os.path.join(category_catalog, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    time_list.append(stats["czas_wykonania"])
                    tokens_in_list.append(stats["zuzycie_tokenow"]["wejsciowe"])
                    tokens_out_list.append(stats["zuzycie_tokenow"]["wyjsciowe"])
                    tokens_all_list.append(stats["zuzycie_tokenow"]["suma"])
                    tool_use += stats["uzycie_narzedzi"]
                    python_use += stats["uzycie_python"]
                    brave_use += stats["uzycie_brave"]
                    wolfram_use += stats["uzycie_wolfram"]
                    rag_use += stats["uzycie_rag"]
                    if stats["uzycie_narzedzi"] != 0:
                        tasks_with_tool_use += 1
            # print(f"Odczytano plik: {filename}")
        all_tasks = right + wrong + error
        # print(f"Ilość rozwiązanych zadań: {all_tasks}")
        # print(f"Ilość zadań z poprawnym wynikiem: {right}")
        # print(f"Ilość zadań z niepoprawnym wynikiem: {wrong}")
        # print(f"Ilość przebiegów zakończonych błędem wykonania: {error}")
        # print(f"Ilość wywołań narzędzi: {tool_use}")
        # print(f"Ile razy rag dodał kontekst: {rag_use}")
        # print(f"Errors %: {100 * error/all_tasks}")
        # print(f"Dokładność: {(100 * right/all_tasks): .2f}")
        # print(f"Mediana czasu na zadanie: {statistics.median(time_list):.2f} s")
        # print(f"Średni czas na zadanie:   {statistics.mean(time_list):.2f} s")
        # print(f"Średnio tokenów IN:       {statistics.mean(tokens_in_list):.0f}")
        # print(f"Średnio tokenów OUT:      {statistics.mean(tokens_out_list):.0f}")
        # print(f"Średnio tokenów ALL:      {statistics.mean(tokens_all_list):.0f}")

        logs.append(
            f"Ilość rozwiązanych zadań: {all_tasks}\n"
            f"Ilość zadań z poprawnym wynikiem: {right}\n"
            f"Ilość zadań z niepoprawnym wynikiem: {wrong}\n"
            f"Ilość przebiegów zakończonych błędem wykonania: {error}\n"
            f"Errors %: {100 * error / all_tasks}\n"
            f"Dokładność: {(100 * right / all_tasks): .2f}\n"
            f"Mediana czasu na zadanie: {statistics.median(time_list):.2f} s\n"
            f"Średni czas na zadanie:   {statistics.mean(time_list):.2f} s\n"
            f"Średnio tokenów IN:       {statistics.mean(tokens_in_list):.0f}\n"
            f"Średnio tokenów OUT:      {statistics.mean(tokens_out_list):.0f}\n"
            f"Średnio tokenów ALL:      {statistics.mean(tokens_all_list):.0f}\n"
            f"Ilość rozwiązań z użyciem narzędzi: {tasks_with_tool_use}\n"
            f"Ilość wywołań narzędzi(ogólnie): {tool_use}\n"
            f"Ilość wywołań pythona: {python_use}\n"
            f"Ilość wywołań wolfram: {wolfram_use}\n"
            f"Ilość wywołań brave: {brave_use}\n"
            f"Ile razy rag dodał kontekst: {rag_use}\n"
        )
        for log in logs:
            print(log, end="")

        with open(os.path.join(self.results_path, 'test.log'), "a", encoding="utf-8") as plik:
            for log in logs:
                plik.write(log + "\n")

    async def evaluate_dataset(self, dataset: Dataset, verifier: bool = False):
        self.results_path = f"results/{self.graph.pipeline_name}/{self.graph.model_name}/{dataset}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_path, exist_ok=True)

        match dataset:
            case Dataset.MATH480:
                for category in sorted(os.listdir("MATH480")):
                    category_dir = os.path.join("MATH480", category)
                    await self.evaluate_llm(category_dir, verifier=verifier)
            case Dataset.AIME:
                await self.evaluate_llm("aime_2025_2026", verifier=verifier)

    def __init__(self, graph: Graph):
        self.graph = graph
        self.results_path = None
