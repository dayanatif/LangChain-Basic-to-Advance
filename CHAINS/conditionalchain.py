from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the review")

parser2 = PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following review: {text} \n {format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {text}',
    input_variables=['text']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "No response needed")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'text': 'This is a beautiful phone'}))
