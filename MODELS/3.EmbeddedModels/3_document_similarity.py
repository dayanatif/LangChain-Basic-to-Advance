from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=10)

documents = [
    "Lewis Hamilton is a British Formula 1 driver known for his speed, consistency, and record-equalling world championships.",
    "Max Verstappen is a Dutch driver famous for his aggressive driving style and multiple world titles.",
    "Sebastian Vettel is a German driver recognized for his four consecutive world championships and sportsmanship.",
    "Charles Leclerc is a Monégasque driver known for his qualifying pace and resilience under pressure.",
    "Fernando Alonso is a Spanish driver celebrated for his strategic brilliance and longevity in Formula 1."
]

query = "Who is the fastest driver in Formula 1?"
query_embedding = embeddings.embed_query(query)
doc_embeddings = embeddings.embed_documents(documents)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)