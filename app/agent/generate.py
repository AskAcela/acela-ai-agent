from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent import llm, llm_with_tools
from app.agent.http_tool import http_request
from app.logger import logger


# ---------------------------------------------------------------------------
# Mode-specific system prompts
# ---------------------------------------------------------------------------

ASK_PREAMBLE = """You are Acela (ah-sell-ah), a sharp and knowledgeable guide to the Celo blockchain ecosystem. \
You speak with quiet confidence — direct, clear, never condescending. \
You care about getting things right, so you stick to what the provided context supports and say so \
when something is outside your knowledge rather than guessing. \
Keep answers concise (3–5 sentences unless depth is genuinely needed). \
Use the HTTP tool only when the context falls short and a specific URL would fill the gap. \
Format your response in clear markdown (headers, bullet points, code blocks) where it improves readability."""

IDEA_PREAMBLE = """You are Acela (ah-sell-ah), a creative strategist for the Celo ecosystem with a builder's mindset. \
You're enthusiastic without being over the top — you get excited about what's possible on Celo \
and that energy comes through in how you write. \
Use the provided context as a launchpad: connect dots, propose angles the user may not have considered, \
and challenge assumptions where it helps. Favour concrete ideas over vague inspiration. \
Use the HTTP tool to pull in a project page, repo, or article when it would spark something tangible. \
Format your response in clear markdown (headers, bullet points, code blocks) where it improves readability."""

EXPLORE_PREAMBLE = """You are Acela (ah-sell-ah), a meticulous researcher of the Celo blockchain ecosystem. \
You're thorough and intellectually honest — you follow threads wherever they lead, \
acknowledge uncertainty when it exists, and cite every source you draw from. \
Use the provided context as a starting point, then use the HTTP tool actively to fetch \
documentation, APIs, GitHub repos, or relevant URLs that deepen your answer. \
Don't stop at one tool call if more would meaningfully improve the response. \
Write with the detail and rigour of someone who will be held accountable for their conclusions. \
Format your response in clear markdown (headers, bullet points, code blocks) where it improves readability."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_generate_node(preamble: str, max_tool_iterations: int):
    """Return a generate node function configured for a specific mode."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", preamble),
            ("placeholder", "{messages}"),
            ("human", "Context:\n{context}"),
        ]
    )

    tools_map = {http_request.name: http_request}

    def generate(state):
        messages = state["messages"]
        documents = state.get("documents", [])
        total_tokens = state.get("total_tokens", 0)

        if not isinstance(documents, list):
            documents = [documents]

        context = "\n\n---\n\n".join(
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in documents
        ) or "No context available."

        # Build the initial message list for this generation step
        gen_messages = prompt.format_messages(messages=messages, context=context)

        generation = None
        for iteration in range(max_tool_iterations):
            response = llm_with_tools.invoke(gen_messages)
            total_tokens += (response.usage_metadata or {}).get("total_tokens", 0)

            if not response.tool_calls:
                generation = response
                break

            logger.info(
                f"Generate node: executing {len(response.tool_calls)} tool call(s) "
                f"(iteration {iteration + 1}/{max_tool_iterations})"
            )

            # Append AI message with tool calls, then execute each tool
            gen_messages.append(response)
            for tc in response.tool_calls:
                tool_fn = tools_map.get(tc["name"])
                result = tool_fn.invoke(tc["args"]) if tool_fn else f"Unknown tool: {tc['name']}"
                logger.info(f"Tool '{tc['name']}' called with args {tc['args']}")
                gen_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        if generation is None:
            # Max iterations reached — force a final answer without tools
            gen_messages.append(
                HumanMessage(content="Based on everything gathered, provide your final answer now.")
            )
            generation = llm.invoke(gen_messages)
            total_tokens += (generation.usage_metadata or {}).get("total_tokens", 0)

        logger.info("Generate node: response ready.")
        return {
            "generation": generation,
            "documents": documents,
            "messages": messages,
            "total_tokens": total_tokens,
        }

    return generate


# ---------------------------------------------------------------------------
# Mode nodes
# ---------------------------------------------------------------------------

generate_ask = make_generate_node(ASK_PREAMBLE, max_tool_iterations=3)
generate_idea = make_generate_node(IDEA_PREAMBLE, max_tool_iterations=2)
generate_explore = make_generate_node(EXPLORE_PREAMBLE, max_tool_iterations=8)
