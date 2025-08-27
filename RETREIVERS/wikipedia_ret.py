from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = "Lewis Hamilton"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:\n")
    print(doc.page_content)
    
