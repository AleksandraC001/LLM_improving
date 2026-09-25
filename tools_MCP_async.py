import os
import asyncio
from langchain_core.tools import tool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

from functools import cache, cached_property

with open("API_wolf", "r") as f:
    WOLFRAM_API_KEY = f.read().strip()
with open("API_brave_search", "r") as f:
    BRAVE_API_KEY = f.read().strip()
my_env = os.environ.copy()
my_env["BRAVE_API_KEY"] = BRAVE_API_KEY


@cache
def python_interpreter():
    print("Łączenie z serwerem MCP Python Interpreter...")

    mcp_python_client = BasicMCPClient(
        command_or_url="timeout",
        args=[
            "60",
            "docker",
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
                    return await t.acall(code=code)
                except Exception as e:
                    return f"Error inside Docker container (Python Interpreter): {str(e)}. Fix tmain()he code and try again."

        return "Error: Python execution tool not found on the Docker MCP server."

    return python_interpreter

@cache
def ask_wolfram():
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
            if "wolfram" in t.metadata.name.lower() or "query" in t.metadata.name.lower():
                try:
                    response = await t.acall(query=query)
                    print(f"\nAGENT PYTA WOLFRAM ALPHA: {query}\n")
                    print(f"odpowiedź wolframa:{response}")
                    return response

                except Exception as e:
                    return f"Error querying Wolfram Alpha: {str(e)}. Try a different tool or syntax."

        return "Error: Wolfram Alpha tool not found on the MCP server."

    return ask_wolfram

@cache
def brave_search():
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
                    response = await t.acall(query=query)
                    print(f"odpowiedź narzędzia brave: {response.content}")
                    return response.content
                except Exception as e:
                    return f"Error inside Docker container (Brave Search): {str(e)}"

        return "Error: Tool brave_web_search not found on the Docker MCP server."

    return brave_search

def get_tools_auto():
    return [python_interpreter(), brave_search(), ask_wolfram()]

# @cache
# def submit_to_verifier():
#     @tool
#     async def submit_to_verifier(final_answer_in_latex: str):
#         """
#         Call this tool ONLY when you have completed all calculations and want to submit your final answer.
#         Provide the exact answer as the argument.
#         """
#         pass
#
#     return submit_to_verifier

# @cache
# def create_thought():
#     @tool
#     async def create_thought(thought: str):
#         """
#         Use this tool to formulate your own logical steps, break down the problem,
#         plan your approach, or analyze the output of previous tools.
#         Call this tool when you need to "think out loud", make deductions,
#         or transition between steps without relying on external computations.
#         """
#         return "Thought recorded. Proceed to the next step."
#
#     return create_thought

# def get_tools_required():
#     return [python_interpreter(), brave_search(), submit_to_verifier(), ask_wolfram()]
