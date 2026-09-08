import httpx
import openai
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END
from langchain_openai import ChatOpenAI
import prompts_used

llm = ChatOpenAI(
    model="nvidia/Llama-3.3-70B-Instruct-NVFP4",
    api_key="empty",
    base_url="http://localhost:8001/v1",
    temperature=0,
    max_tokens=2000,
    timeout=120.0,
    max_retries=2
)

def create_conversation(messages):
    conversation_transcript = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_transcript += f"USER QUESTION: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                conversation_transcript += f"SOLVER (TOOL ATTEMPT):\n{msg.tool_calls}\n"
            else:
                conversation_transcript += f"SOLVER: {msg.content}\n\n"
        elif isinstance(msg, ToolMessage):
            conversation_transcript += f"SYSTEM (OUTPUT): {msg.content}\n\n"
    return conversation_transcript

class NewVerificationResult(BaseModel):
    is_correct: bool = Field(
        description="True ONLY if the solution perfectly meets all 5 criteria. False if there is ANY flaw, hallucination, or unhandled tool error.")
    answer: str = Field(
        description="If is_correct is True: provide the exact final answer from the solution in \\boxed{final_answer} format. If is_correct is False: return an empty string ''.")
    feedback: str = Field(
        description="If is_correct is False: State the exact fail reason and provide a clear instruction for Agent Solver on what to fix. If is_correct is True: return an empty string ''.")


async def verifier(state: dict):
    print("--- WERYFIKATOR ---")
    new_iteration = state.get("iterations", 0) + 1
    if new_iteration >= 3:
        to_end_flag = True
        msg = [SystemMessage(content="Verification limit")]
        return {"messages": msg, "to_end": to_end_flag, "iterations": new_iteration}

    print(f"Rozpoczynam {new_iteration} iterację weryfikatora")
    messages = state["messages"]
    verifier_prompt = prompts_used.new_get_verifier_prompt(create_conversation(messages))
    structured_llm = llm.with_structured_output(NewVerificationResult)

    try:
        print("Weryfikator myśli...\n")
        response = await structured_llm.ainvoke([HumanMessage(content=verifier_prompt)])
        print("WERYFIKATOR:\n")
        print(response)
        to_end_flag = response.is_correct
        if response.is_correct:
            answer = response.answer
        else:
            answer = response.feedback
    except (httpx.ReadTimeout, openai.APITimeoutError):
        print("\n!!! WERYFIKATOR TIMEOUT: Zwracam sztuczny komunikat błędu !!!\n")
        answer = "VERDICT: FEEDBACK: AWARIA WERYFIKACJI - Weryfikator uległ awarii z powodu zbyt długiego czasu oczekiwania na odpowiedź (Timeout)."
        to_end_flag = True

    return {"messages": answer, "to_end": to_end_flag, "iterations": new_iteration}


def verifier_router(state: dict):
    return END if state["to_end"] else "solver"