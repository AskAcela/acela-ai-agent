from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.http_tool import http_request

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([http_request])