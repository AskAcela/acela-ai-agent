from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

def content_to_text(content) -> str:
    """Normalize an LLM message's ``content`` to a plain string.

    Gemini (and other providers) may return ``content`` as a list of parts —
    either plain strings or ``{"type": "text", "text": ...}`` dicts — rather
    than a flat string. Call this before using any string method on it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def convert_messages(messages):

    result = []

    for msg in messages:

        if msg.role == "user" or msg.role == "human":
            result.append(
                HumanMessage(content=msg.content)
            )

        elif msg.role == "assistant" or msg.role == "ai":
            result.append(
                AIMessage(content=msg.content)
            )

        elif msg.role == "tool":
            result.append(
                ToolMessage(content=msg.content)
            )

        elif msg.role == "system":
            result.append(
                SystemMessage(content=msg.content)
            )

    return result