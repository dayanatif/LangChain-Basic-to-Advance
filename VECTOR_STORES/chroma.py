from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Babar Azam is Pakistan's all-format captain and one of the most consistent batsmen in world cricket. Known for his elegant cover drives and ability to chase under pressure, he is considered among the best in modern cricket.",
        metadata={"team": "Pakistan National Team"},
        id="babar_azam"
    ),
    Document(
        page_content="Shaheen Shah Afridi is Pakistan's premier fast bowler. Renowned for his lethal yorkers and ability to swing the new ball, he has been a match-winner in ICC tournaments and leagues around the world.",
        metadata={"team": "Pakistan National Team"},
        id="shaheen_afridi"
    ),
    Document(
        page_content="Mohammad Rizwan, Pakistan's wicketkeeper-batsman, is known for his consistency in T20 cricket. He has formed one of the most successful opening pairs with Babar Azam and is praised for his fitness and work ethic.",
        metadata={"team": "Pakistan National Team"},
        id="mohammad_rizwan"
    ),
    Document(
        page_content="Shadab Khan is Pakistan's leading all-rounder in white-ball cricket. A leg-spinner and aggressive middle-order batsman, he provides balance to the team with his versatility.",
        metadata={"team": "Pakistan National Team"},
        id="shadab_khan"
    ),
    Document(
        page_content="Fakhar Zaman is an explosive opening batsman known for his fearless approach. He holds the record for being the first Pakistani to score a double century in ODIs.",
        metadata={"team": "Pakistan National Team"},
        id="fakhar_zaman"
    )
]

# Initialize Chroma DB
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory="pak_cricket_chroma_db",
    collection_name="pak_players"
)

# Add documents (with fixed IDs)
vector_store.add_documents(docs)

# ✅ View documents (correct way: ids are always returned automatically)
result = vector_store.get(include=["documents", "metadatas"])
print("IDs:", result["ids"])
print("Docs:", result["documents"])
print("Metadata:", result["metadatas"])

# 🔍 Search documents
print("\n--- Similarity Search ---")
print(vector_store.similarity_search(query="Who among these is a bowler?", k=2))

# 🔍 Search with similarity score
print("\n--- Similarity Search With Score ---")
print(vector_store.similarity_search_with_score(query="Who among these is a bowler?", k=2))

# 🔍 Metadata filtering
print("\n--- Filter by team ---")
print(vector_store.similarity_search_with_score(query="batsman", filter={"team": "Pakistan National Team"}))

# ✏️ Update document (Babar Azam)
updated_doc1 = Document(
    page_content="Babar Azam, captain of Pakistan, is widely regarded as one of the top batsmen in the world. He has consistently been among the leading run scorers in ICC tournaments and bilateral series. His calm temperament, elegant stroke play, and leadership have made him the backbone of Pakistan's batting lineup.",
    metadata={"team": "Pakistan National Team"},
    id="babar_azam"
)
vector_store.update_document(document_id="babar_azam", document=updated_doc1)

# ✅ View after update
updated_result = vector_store.get(include=["documents", "metadatas"])
print("\n--- After Update ---")
print(updated_result["ids"])
print(updated_result["documents"])

# ❌ Delete document (Babar Azam)
vector_store.delete(ids=["babar_azam"])

# ✅ View after delete
final_result = vector_store.get(include=["documents", "metadatas"])
print("\n--- After Delete ---")
print(final_result["ids"])
print(final_result["documents"])
