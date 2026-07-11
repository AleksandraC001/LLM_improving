import os
# przetestować te MCP, chyba działają już, ale sam podstawowy plik jest zły, trzeba przeanalizować prompt, logikę, czemu nie używa MCP toolsów, jak działa przepływ dokładnie, jak się zapisuje historia i rozmowa
# jeszcze dosprawdzić implementację
# potem zająć się agentem RAG, promptami i przepływem
# potem zaplanować testy, sprawdzić czy to asynchroniczne ma sens
# potem napisać maila i sie zapytać czy to tak ma być i czy testy
# z takimi wartościami i na takich zbiorach są dobrym kierunkiem i może szkielet pracy.
import asyncio
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# ==========================================
# 1. NARZĘDZIA LOKALNE (Natywne LangChain)
# ==========================================

python_tool = PythonREPLTool()


@tool
def python_interpreter(code: str):
    """
    Executes Python code and returns the result (stdout).
    Use this tool to perform necessary calculations and algebraic manipulations.
    """
    print("\n" + "=" * 40)
    print("AGENT URUCHAMIA KOD PYTHON:")
    print(code)
    print("=" * 40 + "\n")

    result = python_tool.run(code)

    if not result or not result.strip():
        result = "Error: No output generated. You MUST use print() to output the final calculated values so I can see them."

    if len(result) > 2000:
        result = result[:2000] + "\n\n...[Error: The code was cutted because it's too long]"

    print("\n" + "=" * 40)
    print(f" Wynik z Pythona: {result}")
    print("=" * 40 + "\n")

    return result


@tool
def submit_to_verifier(final_answer_in_latex: str):
    """
    Call this tool ONLY when you have completed all calculations and want to submit your final answer.
    Provide the exact answer as the argument.
    """
    pass


# ==========================================
# 2. KONFIGURACJA SERWERÓW MCP
# ==========================================

def read_key(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Ostrzeżenie: Nie znaleziono pliku z kluczem: {filename}")
        return ""


WOLFRAM_API_KEY = read_key("wolf_API")
BRAVE_API_KEY = read_key("brave_search_api")

print("Konfiguracja serwerów MCP (Wolfram, Arxiv, Brave) z użyciem langchain-mcp-adapters...")

# Zamiast łączyć się krok po kroku, tworzymy jeden słownik konfiguracyjny
server_config = {
    "arxiv": {
        "transport": "stdio",
        "command": "docker",
        "args": ["run", "-i", "--rm", "mcp/arxiv-mcp-server"]
    }
}

if WOLFRAM_API_KEY:
    server_config["wolfram"] = {
        "transport": "stdio",
        "command": "docker",
        "args": ["run", "-i", "--rm", "-e", f"WOLFRAM_API_KEY={WOLFRAM_API_KEY}", "mcp/wolfram-alpha"]
    }

if BRAVE_API_KEY:
    server_config["brave"] = {
        "transport": "stdio",
        "command": "docker",
        "args": ["run", "-i", "--rm", "-e", f"BRAVE_API_KEY={BRAVE_API_KEY}", "mcp/brave-search"]
    }


# ==========================================
# 3. POBRANIE I MODYFIKACJA NARZĘDZI MCP
# ==========================================

async def load_mcp_tools():
    # Inicjalizujemy klienta z pełnym słownikiem konfiguracyjnym
    mcp_client = MultiServerMCPClient(server_config)

    # Pobieramy narzędzia (to asynchroniczne, bo wysyła zapytania do kontenerów po stdio)
    raw_mcp_tools = await mcp_client.get_tools()

    # Nadpisanie promptów z MCP, aby były dokładniejsze do Twojego systemu:
    for t in raw_mcp_tools:
        name_lower = t.name.lower()

        if "wolfram" in name_lower or "ask" in name_lower:
            t.description = (
                "Queries the Wolfram Alpha engine to compute results and solve complex mathematical problems. "
                "Use this tool for advanced symbolic algebra, systems of equations, difficult integrals, "
                "retrieving physical or scientific properties, or when standard Python execution is unreliable."
            )
        elif "arxiv" in name_lower or "search_papers" in name_lower:
            t.description = (
                "Searches for scientific papers and abstracts in the Arxiv database. "
                "Always use this tool before solving mathematical problems that require advanced theoretical knowledge."
            )
        elif "brave" in name_lower:
            t.description = (
                "Searches for information on the internet using the Brave search engine. "
                "Use this to find general mathematical definitions, formulas, theorems, "
                "or information that does not require browsing full scientific articles."
            )

    return raw_mcp_tools


# ==========================================
# 4. EKSPORT FINALNEJ LISTY NARZĘDZI
# ==========================================

# Synchronizujemy pobieranie narzędzi, aby plik można było zaimportować z zewnątrz
fetched_mcp_tools = asyncio.run(load_mcp_tools())

tools = [python_interpreter, submit_to_verifier] + fetched_mcp_tools

print(f"Zakończono ładowanie. Dostępne narzędzia dla Solvera: {[t.name for t in tools]}")