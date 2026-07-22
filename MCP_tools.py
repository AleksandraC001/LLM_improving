import os
import asyncio
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


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
        result = result[
                 :2000] + "\n\n...[Error: The code was cutted because it's too long"

    print("\n" + "=" * 40)
    print(f" Wynik z Pythona: {result}")
    print("=" * 40 + "\n")

    return result

# --- INICJALIZACJA MCP WOLFRAM ALPHA ---

print("Łączenie z serwerem MCP Wolfram Alpha...")
import asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from langchain_core.tools import tool

# 1. Wczytanie Twojego klucza
with open("wolf_API", "r") as f:
    WOLFRAM_API_KEY = f.read().strip()

print("Łączenie z serwerem MCP wewnątrz kontenera Docker...")

# 2. Architektura oparta na Docker MCP Toolkit
mcp_wolfram_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",                                       # Tryb interaktywny (stdio) - KRYTYCZNE dla MCP!
        "--rm",                                     # Usuwa kontener z pamięci po zakończeniu pracy agenta
        "-e", f"WOLFRAM_API_KEY={WOLFRAM_API_KEY}", # Wstrzyknięcie klucza do wnętrza kontenera
        "mcp/wolfram-alpha"                         # Nazwa obrazu z Docker Hub
    ]
)

mcp_wolfram_tool_spec = McpToolSpec(client=mcp_wolfram_client)
mcp_wolfram_tools = asyncio.run(mcp_wolfram_tool_spec.to_tool_list_async())
#Wrapper dla wolframa
@tool
def ask_wolfram(query: str):
    """
    Queries the Wolfram Alpha engine to compute results and solve complex mathematical problems.
    Use this tool for advanced symbolic algebra, systems of equations, difficult integrals,
    retrieving physical or scientific properties, or when standard Python execution is unreliable.
    """
    print(f"\nAGENT PYTA WOLFRAM ALPHA: {query}\n")

    for t in mcp_wolfram_tools:
        if "wolfram" in t.metadata.name.lower() or "query" in t.metadata.name.lower():
            try:
                response = t(query=query)
            except Exception:
                try:
                    response = t(input=query)
                except Exception as e:
                    return f"Error querying Wolfram Alpha: {str(e)}"
            return response.content

    return "Error: Wolfram Alpha tool not found on the MCP server."


print("Łączenie z serwerem MCP Arxiv wewnątrz kontenera Docker...")

mcp_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",                    # Tryb interaktywny (niezbędne do komunikacji stdio z MCP)
        "--rm",                  # Usunięcie kontenera z pamięci zaraz po wykonaniu zadania
        "mcp/arxiv-mcp-server"   # Nazwa oficjalnego obrazu z Docker Hub
    ]
)
mcp_tool_spec = McpToolSpec(client=mcp_client)
mcp_llama_tools = asyncio.run(mcp_tool_spec.to_tool_list_async())

# --- NOWY BEZPIECZNY ADAPTER (WRAPPER) DLA ARXIV ---
@tool
def search_arxiv(query: str):
    """
    Searches for scientific papers and abstracts in the Arxiv database.
    Always use this tool before solving mathematical problems that require advanced theoretical knowledge.
    """
    print(f"\nAGENT SZUKA W ARXIV: {query}\n")

    # Przeszukujemy listę narzędzi pobranych w tle przez LlamaIndex
    for t in mcp_llama_tools:
        if t.metadata.name == "search_papers" or "arxiv" in t.metadata.name.lower():
            try:
                # Wywołujemy oryginalne narzędzie MCP
                response = t(query=query)
                return response.content
            except Exception as e:
                # Wychwytujemy błędy z wnętrza Dockera
                return f"Error inside Docker container (Arxiv): {str(e)}"

    return "Error: Tool search_papers not found on the Docker MCP server."


# --- INICJALIZACJA MCP BRAVE SEARCH ---
with open("brave_search_api", "r") as f:
    BRAVE_API_KEY= f.read().strip()

my_env = os.environ.copy()
my_env["BRAVE_API_KEY"] = BRAVE_API_KEY

print("Łączenie z serwerem MCP Brave Search...")
mcp_brave_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",                                    # Tryb interaktywny (stdio)
        "--rm",                                  # Usunięcie kontenera z pamięci po wykonaniu zadania
        "-e", f"BRAVE_API_KEY={BRAVE_API_KEY}",  # Wstrzyknięcie klucza API do wnętrza kontenera
        "mcp/brave-search"                       # Oficjalny obraz z Docker Hub
    ]
)
mcp_brave_tool_spec = McpToolSpec(client=mcp_brave_client)
mcp_brave_tools = asyncio.run(mcp_brave_tool_spec.to_tool_list_async())

# --- NOWY BEZPIECZNY ADAPTER (WRAPPER) DLA BRAVE SEARCH ---
@tool
def brave_search(query: str):
    """
    Searches for information on the internet using the Brave search engine.
    Use this to find general mathematical definitions, formulas, theorems,
    or information that does not require browsing full scientific articles.
    """
    print(f"\nAGENT SZUKA W BRAVE: {query}\n")

    for t in mcp_brave_tools:
        # Szukamy narzędzia na serwerze MCP
        if t.metadata.name == "brave_web_search" or "brave" in t.metadata.name.lower():
            try:
                # Wywołanie narzędzia wewnątrz kontenera
                response = t(query=query)
                return response.content
            except Exception as e:
                # Wychwytujemy błędy (np. brak internetu, zły klucz)
                return f"Error inside Docker container (Brave Search): {str(e)}"

    return "Error: Tool brave_web_search not found on the Docker MCP server."

@tool
def submit_to_verifier(final_answer_in_latex: str):
    """
    Call this tool ONLY when you have completed all calculations and want to submit your final answer.
    Provide the exact answer as the argument.
    """
    pass # To narzędzie nigdy nie wykonuje kodu w Pythonie. Router je przechwyci.

# @tool
# def return_final_answer(final_answer_in_latex: str):
#     """
#     Call this tool ONLY when you have completed all calculations and want to submit your final answer.
#     Provide the exact answer as the argument.
#     """
#     pass # To narzędzie nigdy nie wykonuje kodu w Pythonie. Router je przechwyci.

tools = [python_interpreter, search_arxiv, brave_search, submit_to_verifier, ask_wolfram]