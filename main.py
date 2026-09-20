from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_browser, wikipedia

load_dotenv()

MAX_ITERATIONS = 3

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

reserach_llm = ChatGoogleGenerativeAI(
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


class CritiqueResponse(BaseModel):
    is_correct: bool
    errors: list[str]
    suggestions: list[str]

critic_parser = PydanticOutputParser(
    pydantic_object=CritiqueResponse
)

critic_prompt = """
You are a strict research reviewer.

Review the research response given to you.

Check for:
- factual errors
- missing important information
- unsupported claims
- incorrect sources
- contradictions

Return:
- is_correct: true if the response is satisfactory
- errors: list of problems
- suggestions: list of improvements
Format your response in the following format:
{format_instructions}
""".format(format_instructions=critic_parser.get_format_instructions())

critic_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_output_tokens=1024
)


tools = [search_browser, wikipedia]
research_agent = create_agent(
    model=reserach_llm,
    tools=tools,
    system_prompt=prompt
)

critic_agent = create_agent(
    model=critic_llm,
    tools=tools,
    system_prompt=critic_prompt
)


# QUERY

query = input("Enter a topic to research: ")

research_result = None
critique_result = None


for iteration in range(MAX_ITERATIONS):

    print(f"\nIteration {iteration + 1} of {MAX_ITERATIONS}")

    if research_result is None:
        research_instruction = query
    else:
        research_instruction = f"""
Original research question:
{query}

Previous research:
{research_result.model_dump_json(indent=2)}

Previous critique:
{critique_result.model_dump_json(indent=2)}

Improve the previous research using the critic's feedback.

Return a new corrected research response.
"""


    response = research_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": research_instruction
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

    content = (
        content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    research = parser.parse(content)
    research_result = research


    print("\nResearch:")
    print(research.model_dump_json(indent=2))


    # CRITIC
    critique_response = critic_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
Original research question:
{query}

Research response:
{research.model_dump_json(indent=2)}

Review this research against the original question.

Check for:
- factual errors
- missing important information
- unsupported claims
- incorrect sources
- contradictions
"""
            }
        ]
    })


    # PARSE CRITIQUE
    critique_content = critique_response["messages"][-1].content

    if isinstance(critique_content, list):
        critique_content = "".join(
            item["text"]
            for item in critique_content
            if item.get("type") == "text"
        )

    critique_content = (
        critique_content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    critique = critic_parser.parse(critique_content)

    critique_result = critique


    print("\nCritique:")
    print(critique.model_dump_json(indent=2))


    # CHECK RESULT
    if critique.is_correct:

        print("\nResearch is correct. Stopping iterations.")
        break

    else:

        print("\nResearch needs improvement. Continuing...")




print("FINAL RESEARCH RESULT")

print(f"\nTopic: {research_result.topic}")

print(f"\nSummary:\n{research_result.summary}")

print("\nSources:")

for source in research_result.sources:
    print(f"- {source}")

print("\nTools Used:")

for tool in research_result.tools_used:
    print(f"- {tool}")