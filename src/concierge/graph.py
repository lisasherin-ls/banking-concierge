"""The Bank of Australia Concierge graph.

"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, PIIMiddleware
from langchain_openai import ChatOpenAI

from concierge.context import get_prompt
from concierge.tools import TOOLS

load_dotenv(override=True)

# The system prompt (AGENTS.md) is pulled from LangSmith Context Hub at module
# import; a hub edit is picked up on the next process start. Falls back to the
# seed in concierge.prompts.SYSTEM_PROMPT when the hub is unreachable.
SYSTEM_PROMPT = get_prompt()


graph = create_agent(
    model=ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
        temperature=0
    ),
    tools=TOOLS,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"transfer_funds": {
                "allowed_decisions": ["approve", "edit", "reject"]}}),
        PIIMiddleware(
            "credit_card", strategy="mask", apply_to_tool_results=True, apply_to_input=True)
        ],
    system_prompt=SYSTEM_PROMPT
)