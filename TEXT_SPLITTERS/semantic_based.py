from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Premier League is the biggest football league in the world. People all over the world watch the matches and cheer for their favourite teams.


Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data and make decisions. From healthcare to finance, AI systems are improving efficiency and accuracy. Researchers are constantly developing new algorithms to solve complex problems, while businesses adopt AI to gain a competitive edge. As AI technology advances, it raises important questions about ethics, privacy, and the future of work.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
