import enum
from dataclasses import dataclass

class Model(enum.StrEnum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    LLAMA = "llama-3.3-70B-Instruct"
    GEMMA = "gemma-4-26B-A4B"


@dataclass
class ModelParams:
    max_tokens: int
    timeout: float
    max_retries: int

llama_params1 = ModelParams(2000, 900.0, 0)
llama_params2 = ModelParams(2000, 240.0, 2)
