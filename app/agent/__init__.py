from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.http_tool import http_request

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
llm_with_tools = llm.bind_tools([http_request])