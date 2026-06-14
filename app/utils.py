from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

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