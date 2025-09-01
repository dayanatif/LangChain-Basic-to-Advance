from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_function(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=8b90520f64aa598c869bd25c4485178c&query={city}'

  response = requests.get(url)

  return response.json()


llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

prompt = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm = llm,
    tools = [search_tool, get_weather_function],
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool, get_weather_function],
    verbose = True
)

response = agent_executor.invoke({"input": "Find the capital of Pakistan, then find it's current weather condition"})

print(response['output'])