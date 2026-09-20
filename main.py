from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_browser, wikipedia

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=1024
)


parser = PydanticOutputParser(
    pydantic_object=ResearchResponse
)


prompt = """
You are a research assistant.

You will be given a topic and you will provide a summary
of the topic along with sources and tools used for research.

Answer in the following format:

{format_instructions}

Use only the necessary tools and sources to provide
a concise summary of the topic.
""".format(
    format_instructions=parser.get_format_instructions()
)

tools = [search_browser, wikipedia]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt
)


query = input("Enter a topic to research: ")


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})

content = response["messages"][-1].content
if isinstance(content, list):
    content = "".join(
        item["text"]
        for item in content
        if item.get("type") == "text"
    )
content = content.replace("```json", "").replace("```", "").strip()
research = parser.parse(content)

print(research)