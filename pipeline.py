from enum import StrEnum, auto

from graph_builders import BaselineBuilder, McpBuilder, RagBuilder, RagWithMcpAndVerifierBuilder, RagWithMcpBuilder


class Pipeline(StrEnum):
    BASELINE = auto()
    RAG = auto()
    MCP = auto()
    RAG_WITH_MCP = auto()
    RAG_WITH_MCP_AND_VERIFIER = auto()

pipeline_to_builder_module = {
    Pipeline.BASELINE: BaselineBuilder,
    Pipeline.RAG: RagBuilder,
    Pipeline.MCP: McpBuilder,
    Pipeline.RAG_WITH_MCP: RagWithMcpBuilder,
    Pipeline.RAG_WITH_MCP_AND_VERIFIER: RagWithMcpAndVerifierBuilder,
}
