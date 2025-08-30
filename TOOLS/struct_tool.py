from langchain.tools import StructuredTool
from pydantic import Field, BaseModel

class MultiplyInput(BaseModel):
    a: int = Field(required = True, description="The first number to multiply.")
    b: int = Field(required = True, description="The second number to multiply.")

def mult_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=mult_func,
    name = 'Multiplier',
    description = 'Multiplies two numbers.',
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a": 3, "b": 5})
print(result)