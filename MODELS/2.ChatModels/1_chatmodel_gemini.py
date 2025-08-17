##This code demonstrates how to use the Google Gemini chat model with LangChain.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = model.invoke("What is the best way to learn Python?")
print(response.content)