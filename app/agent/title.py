from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Separate LLM so this can be swapped for a smaller/cheaper model independently
_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a short conversation title (4 words maximum) based on the user's first message. "
            "Return only the title — no punctuation, no quotes, no explanation.",
        ),
        ("human", "{message}"),
    ]
)

_chain = _prompt | _llm | StrOutputParser()


def generate_title(message: str) -> str:
    return _chain.invoke({"message": message})
