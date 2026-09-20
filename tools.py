from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool


search = DuckDuckGoSearchResults()

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


@tool
def search_browser(query: str) -> str:
    """Search the internet for current events and factual information."""
    return search.run(query)


@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for general factual information."""
    return wiki.run(query)