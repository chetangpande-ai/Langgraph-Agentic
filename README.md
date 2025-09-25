# LangChain Tools


## Overview
This repository explores **LangChain Tools**, showcasing how they can be used for building and integrating AI-driven applications. The notebook demonstrates practical examples, code snippets, and workflows to help understand LangChain’s tool ecosystem.

```python
from langchain_community.tools import WikipediaQueryRun
```

```python
from langchain_community.utilities import WikipediaAPIWrapper
```

```python
api_wrapper=WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max= 500)
```

```python
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool.name
wiki_tool.description
wiki_tool.args
```

```python
wiki_tool.run({"query":"elon musk"})
```

'Page: Elon Musk\nSummary: Elon Reeve Musk ( EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.\nBorn to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth'

```python
wiki_tool.run("RCB")
```

```python
from langchain_community.tools import YouTubeSearchTool
yt_tool=YouTubeSearchTool()
yt_tool.name
yt_tool.description
yt_tool.args

```

```python
yt_tool.run("github copilot")
```

```python
from langchain_community.tools.tavily_search import TavilySearchResults
import os
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
tavily_tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
tavily_tool.name
tavily_tool.run("H1B news")
```
