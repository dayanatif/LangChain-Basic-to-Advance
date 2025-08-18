## SIMPLE CHAIN

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template="give 5 facts about {topic}",
    input_variables=["topic"]
)

chain = template | model | parser

#CHAIN VISUALIZE
chain.get_graph().print_ascii()

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)