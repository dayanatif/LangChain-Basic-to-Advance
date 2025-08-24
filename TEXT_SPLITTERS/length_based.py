from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20, separator=""

)

loader = PyPDFLoader("example.pdf")
documents = loader.load()
texts = splitter.split_documents(documents)