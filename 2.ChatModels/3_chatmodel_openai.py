## This code demonstrates how to use the OpenAI chat model with LangChain.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write about lewis hamilton in 10 words.")

print(result.content)