import os
import subprocess

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

print(f"\n1.Bazowy Solver \n2. RAG + Solver \n3. Solver + MCP \n4. RAG + Solver + MCP\n5.Weryfikator")
pipeline = input("Wybierz z listy numer przepływu do rozwiązania zadania:")
choose_dataset = input("\n1. MATH480 \n2. AIME2025+AIME2026 \nWybierz z listy numer zbioru danych do ewaluacji: ")
print(f"\n1.GPT-4o \n2.GPT-4o-mini \n3.Llama3.3-70B-Instruct")
model = input("Wybierz numer modelu solvera")
modele = {
    "1": "gpt-4o",
    "2": "gpt-4o-mini",
    "3": "llama-3.3-70B-Instruct"
}

datasets = {
    "1": "MATH480",
    "2": "AIME",
}

if pipeline == "3":
    tool_calling = input("Wybierz parametr wywoływania narzędzi: \n1. auto \n2. requested")

parameters = {
    "1": "auto",
    "2": "required",
}

if pipeline == "1":
    print(f"Rozpoczynam ewaluację bazowego Solvera (Model: {modele.get(model)}) na zbiorze {datasets.get(choose_dataset)}...")
    subprocess.run(["python", "pipeline_baseline.py", "--model", modele.get(model),
    "--dataset", datasets.get(choose_dataset)])
if pipeline == "2":
    print(f"Rozpoczynam ewaluację RAG + Solver (Model: {modele.get(model)}) na zbiorze {datasets.get(choose_dataset)}...")
    subprocess.run(["python", "pipeline_solver+RAG.py",  "--model", modele.get(model), "--dataset", datasets.get(choose_dataset)])
if pipeline == "3":
    print(f"Rozpoczynam ewaluację przepływu Solver + narzędzia MCP (Model: {modele.get(model)}) z parametrem wyboru narzędzi: {parameters.get(tool_calling)} na zbiorze {datasets.get(choose_dataset)}...")
    subprocess.run(["python", "pipeline_solver+MCP_parameter.py", "--model", modele.get(model),
    "--dataset", datasets.get(choose_dataset), "--parameter", parameters.get(tool_calling)])
