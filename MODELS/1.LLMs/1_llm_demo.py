from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

response = llm.invoke("What is the capital of Australia?")
print(response)
