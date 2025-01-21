from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

# web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the model for the information",
    model=Groq(id="gemma2-9b-it"),
    tools=[DuckDuckGo()],
    instructions=["Always include the source."],
    show_tools_calls=True,
    markdown=True,
)

# financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="gemma2-9b-it"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Use tables to display data where possible."],
    markdown=True,
)

app = Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__ =="__main__":
    serve_playground_app("playground:app",reload=True)