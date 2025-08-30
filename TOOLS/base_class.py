from langchain.tools import BaseTool
from typing import Type
from pydantic import Field, BaseModel

class multiply_input(BaseModel):
    a: int = Field(required = True, description="The first number to multiply.")
    b: int = Field(required = True, description="The second number to multiply.")

class multiply(BaseTool):
    name: str = "Multiplier"
    description: str = "Multiplies two numbers."
    args_schema: Type[BaseModel] = multiply_input

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = multiply()

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
