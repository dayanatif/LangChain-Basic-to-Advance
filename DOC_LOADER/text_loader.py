from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

prompt = PromptTemplate(
    template='Summarize the following poem: {text}',
    input_variables=['text']
)

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)