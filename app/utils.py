from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

def convert_messages(messages):

    result = []

    for msg in messages:

        if msg.role == "user":
            result.append(
                HumanMessage(content=msg.content)
            )

        elif msg.role == "assistant":
            result.append(
                AIMessage(content=msg.content)
            )

        elif msg.role == "system":
            result.append(
                SystemMessage(content=msg.content)
            )

    return result