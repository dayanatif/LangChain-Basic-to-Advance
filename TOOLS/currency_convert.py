from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json
from dotenv import load_dotenv

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Get the conversion factor from base_currency to target_currency."""
    response = requests.get(f"https://v6.exchangerate-api.com/v6/0dfd831f1bbb375e5ac7c6b2/pair/{base_currency}/{target_currency}")
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """
  return base_currency_value * conversion_rate


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between USD and PKR, and based on that can you convert 10 USD TO PKR?')]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:

    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        messages.append(tool_message1)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']

    elif tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

print(llm_with_tools.invoke(messages).content)