import os
import asyncio
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

python_tool = PythonREPLTool()


@tool
async def python_interpreter(code: str):
    """
    Executes Python code and returns the result (stdout).
    Use this tool to perform necessary calculations and algebraic manipulations.
    """
    print("\n" + "=" * 40)
    print("AGENT URUCHAMIA KOD PYTHON:")
    print(code)
    print("=" * 40 + "\n")

    result = await asyncio.to_thread(python_tool.run, code)

    if not result or not result.strip():
        result = "Error: No output generated. You MUST use print() to output the final calculated values so I can see them."

    if len(result) > 2000:
        result = result[:2000] + "\n\n...[Error: The code was cutted because it's too long]"

    print("\n" + "=" * 40)
    print(f" Wynik z Pythona: {result}")
    print("=" * 40 + "\n")

    return result


print("Łączenie z serwerem MCP Wolfram Alpha...")

with open("wolf_API", "r") as f:
    WOLFRAM_API_KEY = f.read().strip()

print("Łączenie z serwerem MCP wewnątrz kontenera Docker...")

mcp_wolfram_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e", f"WOLFRAM_API_KEY={WOLFRAM_API_KEY}",
        "mcp/wolfram-alpha"
    ]
)

mcp_wolfram_tool_spec = McpToolSpec(client=mcp_wolfram_client)
mcp_wolfram_tools = asyncio.run(mcp_wolfram_tool_spec.to_tool_list_async())


@tool
async def ask_wolfram(query: str):
    """
    Queries the Wolfram Alpha engine to compute results and solve complex mathematical problems.
    CRITICAL INSTRUCTION: DO NOT use raw LaTeX (e.g. \\sqrt, \\frac, \\root) in your query!
    Always convert mathematical expressions to plain text syntax (e.g. use x^(1/3) instead of \\root 3, and x/y instead of \\frac{x}{y}).
    """
    print(f"\nAGENT PYTA WOLFRAM ALPHA: {query}\n")

    # for t in mcp_wolfram_tools:
    #     if "wolfram" in t.metadata.name.lower() or "query" in t.metadata.name.lower():
    #         try:
    #             response = await asyncio.to_thread(t, query=query)
    #
    #             result_text = ""
    #
    #             # ignorowanie OBRAZKÓW
    #             if hasattr(response, 'content') and isinstance(response.content, list):
    #                 for item in response.content:
    #                     if hasattr(item, 'text') and item.text:
    #                         result_text += item.text + "\n"
    #             else:
    #                 result_text = str(getattr(response, 'content', response))
    #
    #             if not result_text.strip():
    #                 return "Error: Wolfram Alpha returned an empty text response or only images."
    #
    #             if getattr(response, 'isError', False) or "Failed to query" in result_text:
    #                 return (f"ERROR: Wolfram Alpha could not understand the query. "
    #                         f"DO NOT REPEAT THIS EXACT QUERY. "
    #                         f"Rewrite your math expression into standard plain text (e.g., x^(1/3)) or switch to using the 'python_interpreter' tool.")
    #
    #             print(f"odpowiedź narzędzia wolfram: {result_text.strip()}")
    #             return result_text.strip()
    #         except Exception as e:
    #             return f"Error querying Wolfram Alpha: {str(e)}. Try a different tool or syntax."
    #
    # return "Error: Wolfram Alpha tool not found on the MCP server."
import re

@tool
async def ask_wolfram(query: str):
    """
    Queries the Wolfram Alpha engine to compute results and solve complex mathematical problems.
    CRITICAL INSTRUCTION: DO NOT use raw LaTeX (e.g. \sqrt, \frac, \root) in your query!
    Always convert mathematical expressions to plain text syntax (e.g. use x^(1/3) instead of \root 3).
    """
    print(f"\nAGENT PYTA WOLFRAM ALPHA: {query}\n")
    for t in mcp_wolfram_tools:
        if "wolfram" in t.metadata.name.lower() or "query" in t.metadata.name.lower():
            try:
                response = await asyncio.to_thread(t, query=query)
                result_text = ""
                content_list = getattr(response, 'content', response)
                if isinstance(content_list, list):
                    for item in content_list:
                        item_type = getattr(item, 'type', '')
                        if item_type == 'text':
                            text_val = getattr(item, 'text', '')
                            if text_val:
                                result_text += text_val + "\n"
                        elif hasattr(item, 'text') and item.text and not getattr(item, 'type', '') == 'image':
                            result_text += item.text + "\n"
                else:
                    result_text = str(content_list)

                # Wycina obiekty ImageContent ze stringów
                result_text = re.sub(r"ImageContent\(.*?\)", "", result_text)
                # Wycina długie ciągi Base64
                result_text = re.sub(r"data='[A-Za-z0-9+/=]{50,}'", "data='[USUNIĘTY OBRAZEK]'", result_text)
                # Wycina zduplikowane spacje i entery
                result_text = re.sub(r'\n\s*\n', '\n', result_text).strip()

                if not result_text:
                    return "Error: Wolfram Alpha returned an empty response or only images."

                if getattr(response, 'isError', False) or "Failed to query" in result_text:
                    return (f"ERROR: Wolfram Alpha could not understand the query. "
                            f"DO NOT REPEAT THIS EXACT QUERY. "
                            f"Rewrite your math expression into standard plain text (e.g., x^(1/3)) or switch to using the 'python_interpreter' tool.")

                if len(result_text) > 1500:
                    result_text = result_text[
                                  :1500] + "\n\n...[CRITICAL WARNING: Output was too long and was truncated. Use Python if you need more precise steps]"

                print(f"Oczyszczona odpowiedź Wolfram: {result_text}")
                return result_text

            except Exception as e:
                return f"Error querying Wolfram Alpha: {str(e)}. Try a different tool or syntax."

    return "Error: Wolfram Alpha tool not found on the MCP server."


mcp_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "mcp/arxiv-mcp-server"
    ]
)
mcp_tool_spec = McpToolSpec(client=mcp_client)
mcp_llama_tools = asyncio.run(mcp_tool_spec.to_tool_list_async())


@tool
async def search_arxiv(query: str):
    """
    Searches for scientific papers and abstracts in the Arxiv database.
    Always use this tool before solving mathematical problems that require advanced theoretical knowledge.
    """
    print(f"\nAGENT SZUKA W ARXIV: {query}\n")

    for t in mcp_llama_tools:
        if t.metadata.name == "search_papers" or "arxiv" in t.metadata.name.lower():
            try:
                response = await asyncio.to_thread(t, query=query)
                print(f"odpowiedź narzędzia arxiv: {response.content}")
                return response.content
            except Exception as e:
                return f"Error inside Docker container (Arxiv): {str(e)}"

    return "Error: Tool search_papers not found on the Docker MCP server."


with open("brave_search_api", "r") as f:
    BRAVE_API_KEY = f.read().strip()

my_env = os.environ.copy()
my_env["BRAVE_API_KEY"] = BRAVE_API_KEY

print("Łączenie z serwerem MCP Brave Search...")
mcp_brave_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e", f"BRAVE_API_KEY={BRAVE_API_KEY}",
        "mcp/brave-search"
    ]
)
mcp_brave_tool_spec = McpToolSpec(client=mcp_brave_client)
mcp_brave_tools = asyncio.run(mcp_brave_tool_spec.to_tool_list_async())


@tool
async def brave_search(query: str):
    """
    Searches for information on the internet using the Brave search engine.
    Use this to find general mathematical definitions, formulas, theorems,
    or information that does not require browsing full scientific articles.
    """
    print(f"\nAGENT SZUKA W BRAVE: {query}\n")

    for t in mcp_brave_tools:
        if t.metadata.name == "brave_web_search" or "brave" in t.metadata.name.lower():
            try:
                response = await asyncio.to_thread(t, query=query)
                print(f"odpowiedź narzędzia brave: {response.content}")
                return response.content
            except Exception as e:
                return f"Error inside Docker container (Brave Search): {str(e)}"

    return "Error: Tool brave_web_search not found on the Docker MCP server."


@tool
async def submit_to_verifier(final_answer_in_latex: str):
    """
    Call this tool ONLY when you have completed all calculations and want to submit your final answer.
    Provide the exact answer as the argument.
    """
    pass


tools = [python_interpreter, search_arxiv, brave_search, submit_to_verifier, ask_wolfram]