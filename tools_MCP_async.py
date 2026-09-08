import os
import asyncio
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

python_tool = PythonREPLTool()

with open("API_wolf", "r") as f:
    WOLFRAM_API_KEY = f.read().strip()
with open("API_brave_search", "r") as f:
    BRAVE_API_KEY = f.read().strip()
my_env = os.environ.copy()
my_env["BRAVE_API_KEY"] = BRAVE_API_KEY

# Uruchomienie serwera MCP dla interpretera Pythona
print("Łączenie z serwerem MCP Python Interpreter...")

mcp_python_client = BasicMCPClient(
    command_or_url="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "better-python"
    ]
)

mcp_python_tool_spec = McpToolSpec(client=mcp_python_client)
mcp_python_tools = asyncio.run(mcp_python_tool_spec.to_tool_list_async())



@tool
async def python_interpreter(code: str):
    """
    Executes Python code in an isolated environment and returns the result (stdout).
    Use this tool to perform necessary calculations and algebraic manipulations.
    This environment is STATELESS. Variables, functions, and imports DO NOT persist between executions.
    You must define all necessary variables, re-import modules, and complete your full calculation within a single tool call.
    """

    for t in mcp_python_tools:
        if "python" in t.metadata.name.lower() or "execute" in t.metadata.name.lower():
            try:
                response = await asyncio.to_thread(t, code=code)
                return response

            except Exception as e:
                return f"Error inside Docker container (Python Interpreter): {str(e)}. Fix the code and try again."

    return "Error: Python execution tool not found on the Docker MCP server."


# print("Łączenie z serwerem MCP Wolfram Alpha...")



# print("Łączenie z serwerem MCP wewnątrz kontenera Docker...")

mcp_wolfram_client = BasicMCPClient(
    command_or_url="docker",
    # args = [
    #     "attach",
    #     # "exec",
    #     # "-i",
    #     "better-wolfram",
    #     # "/bin/bash"
    # ]
    args=[
        "run",
        "-i",
        "--rm",
        "-e", f"WOLFRAM_API_KEY={WOLFRAM_API_KEY}",
        "better-wolfram"
    ]
)

mcp_wolfram_tool_spec = McpToolSpec(client=mcp_wolfram_client)
mcp_wolfram_tools = asyncio.run(mcp_wolfram_tool_spec.to_tool_list_async())

import re

@tool
async def ask_wolfram(query: str):
    """
    Queries the Wolfram Alpha engine to compute results and solve complex mathematical problems.
    CRITICAL INSTRUCTIONS:
    - DO NOT use raw LaTeX (e.g. \sqrt, \frac, \root) in your query!
    - DO NOT use natural language, conversational text, or full sentences (e.g., NEVER write "List the positive factors of 20").
    - Use ONLY raw, concise mathematical commands and keywords (e.g., "factors of 20", "solve x^2=4", "integrate sin(x)").
    Always convert mathematical expressions to plain text syntax (e.g. use x^(1/3) instead of \root 3).
    """

    for t in mcp_wolfram_tools:
        #04.09.2026 zakomentowane
        if "wolfram" in t.metadata.name.lower() or "query" in t.metadata.name.lower():
            try:
                response = await asyncio.to_thread(t, query=query)
                print(f"\nAGENT PYTA WOLFRAM ALPHA: {query}\n")
                print(f"odpowiedź wolframa:{response}")
                # result_text = ""
                # content_list = getattr(response, 'content', response)
                # if isinstance(content_list, list):
                #     for item in content_list:
                #         item_type = getattr(item, 'type', '')
                #         if item_type == 'text':
                #             text_val = getattr(item, 'text', '')
                #             if text_val:
                #                 result_text += text_val + "\n"
                #         elif hasattr(item, 'text') and item.text and not getattr(item, 'type', '') == 'image':
                #             result_text += item.text + "\n"
                # else:
                #     result_text = str(content_list)
                #
                # # Wycina obiekty ImageContent ze stringów
                # result_text = re.sub(r"ImageContent\(.*?\)", "", result_text)
                # # Wycina długie ciągi Base64
                # result_text = re.sub(r"data='[A-Za-z0-9+/=]{50,}'", "data='[USUNIĘTY OBRAZEK]'", result_text)
                # # Wycina zduplikowane spacje i entery
                # result_text = re.sub(r'\n\s*\n', '\n', result_text).strip()
                #
                # if not result_text:
                #     return "Error: Wolfram Alpha returned an empty response or only images."
                #
                # if getattr(response, 'isError', False) or "Failed to query" in result_text:
                #     return (f"ERROR: Wolfram Alpha could not understand the query. "
                #             f"DO NOT REPEAT THIS EXACT QUERY. "
                #             f"Rewrite your math expression into standard plain text (e.g., x^(1/3)) or switch to using the 'python_interpreter' tool.")
                #
                # if len(result_text) > 1500:
                #     result_text = result_text[
                #                   :1500] + "\n\n...[CRITICAL WARNING: Output was too long and was truncated. Use Python if you need more precise steps]"
                #
                # print(f"Oczyszczona odpowiedź Wolfram: {result_text}")
                return response

            except Exception as e:
                return f"Error querying Wolfram Alpha: {str(e)}. Try a different tool or syntax."

    return "Error: Wolfram Alpha tool not found on the MCP server."


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
    Use this to find general mathematical definitions, formulas, theorems.
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

@tool
async def create_thought(thought: str):
    """
    Use this tool to formulate your own logical steps, break down the problem,
    plan your approach, or analyze the output of previous tools.
    Call this tool when you need to "think out loud", make deductions,
    or transition between steps without relying on external computations.
    """
    return "Thought recorded. Proceed to the next step."

tools_auto = [python_interpreter, brave_search, ask_wolfram]# search_arxiv,
tools_required = [python_interpreter, brave_search, submit_to_verifier, ask_wolfram]