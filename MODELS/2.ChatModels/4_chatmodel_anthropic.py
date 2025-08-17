### This code demonstrates how to use the Anthropic chat model with LangChain.

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke('What is the meaning of life?')

print(result.content)