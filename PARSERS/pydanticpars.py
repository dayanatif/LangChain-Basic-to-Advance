##PYDANTIC OUTPUT PARSER

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)



class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate name, age and city for a {person} fictional person \n {format_instruction}',
    input_variables=['person'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

parser = PydanticOutputParser(pydantic_object=Person)

chain = template | model | parser

result = chain.invoke({'person':'Pakistani'})

print(result)