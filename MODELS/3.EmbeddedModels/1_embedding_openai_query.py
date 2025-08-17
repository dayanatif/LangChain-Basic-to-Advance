## This code demonstrates how to use OpenAI's embedding model with LangChain to embed a query.

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=30)

query = "What is the color of sky?"

result = embeddings.embed_query(query)
print(str(result))