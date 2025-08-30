from langchain_core.tools import tool

@tool
def multiply(a:int, b:int) -> int:
    """
    Multiplies two numbers.
    """
    return a * b

result = multiply.invoke({"a":3, "b":5})
print(result)
