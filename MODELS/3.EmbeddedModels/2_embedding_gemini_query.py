from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=10)

query = "What is an apple"

result = embeddings.embed_query(query)
print(str(result))