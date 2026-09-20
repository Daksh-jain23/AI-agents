from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ResearchREsponse(BaseModel):
    topic: str
    summary: str
    sources : list[str]
    tools_used: list[str]



llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0.2, 
    max_output_tokens=1024
)


response = llm.invoke("Hi, how are you?")

print(response.content)