from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Create LangChain documents for Pakistani players

doc1 = Document(
    page_content="Babar Azam is Pakistan's all-format captain and one of the most consistent batsmen in world cricket. Known for his elegant cover drives and ability to chase under pressure, he is considered among the best in modern cricket.",
    metadata={"team": "Pakistan National Team"}
)

doc2 = Document(
    page_content="Shaheen Shah Afridi is Pakistan's premier fast bowler. Renowned for his lethal yorkers and ability to swing the new ball, he has been a match-winner in ICC tournaments and leagues around the world.",
    metadata={"team": "Pakistan National Team"}
)

doc3 = Document(
    page_content="Mohammad Rizwan, Pakistan's wicketkeeper-batsman, is known for his consistency in T20 cricket. He has formed one of the most successful opening pairs with Babar Azam and is praised for his fitness and work ethic.",
    metadata={"team": "Pakistan National Team"}
)

doc4 = Document(
    page_content="Shadab Khan is Pakistan's leading all-rounder in white-ball cricket. A leg-spinner and aggressive middle-order batsman, he provides balance to the team with his versatility.",
    metadata={"team": "Pakistan National Team"}
)

doc5 = Document(
    page_content="Fakhar Zaman is an explosive opening batsman known for his fearless approach. He holds the record for being the first Pakistani to score a double century in ODIs.",
    metadata={"team": "Pakistan National Team"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# Initialize Chroma DB
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='pak_cricket_chroma_db',
    collection_name='pak_players'
)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=['embeddings', 'documents', 'metadatas'])

# search documents
vector_store.similarity_search(
    query='Who among these is a bowler?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these is a bowler?',
    k=2
)

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Pakistan National Team"}
)

# update document
updated_doc1 = Document(
    page_content="Babar Azam, captain of Pakistan, is widely regarded as one of the top batsmen in the world. He has consistently been among the leading run scorers in ICC tournaments and bilateral series. His calm temperament, elegant stroke play, and leadership have made him the backbone of Pakistan's batting lineup.",
    metadata={"team": "Pakistan National Team"}
)

# Replace with actual document_id returned by Chroma when you first inserted docs
vector_store.update_document(document_id='PUT-YOUR-DOC-ID-HERE', document=updated_doc1)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# delete document
vector_store.delete(ids=['PUT-YOUR-DOC-ID-HERE'])

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])
