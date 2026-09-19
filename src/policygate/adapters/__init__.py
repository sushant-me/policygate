"""Adapters that put the gate in front of real agent tool machinery.

Each adapter follows the same non-negotiable rule: **if the gate does not allow the
call, the underlying tool is never invoked.** Not invoked-and-logged, not invoked with a
warning — the side effect must not happen, and the tests assert exactly that with a
call counter rather than by trusting the return value.

The adapters are duck-typed on purpose. Importing LangChain or an MCP SDK to gate a call
would make this package unusable in the environments that need it most, so each adapter
asks only for the shape it needs and says so in its docstring.
"""

from .function_calls import execute_function_call, gate_function_call, parse_function_call
from .langchain import GatedTool, gated_tool, gated_tools
from .mcp import RESERVED_TOOL_NAMES, GatedMCPServer

__all__ = [
    "GatedMCPServer",
    "GatedTool",
    "RESERVED_TOOL_NAMES",
    "execute_function_call",
    "gate_function_call",
    "gated_tool",
    "gated_tools",
    "parse_function_call",
]
