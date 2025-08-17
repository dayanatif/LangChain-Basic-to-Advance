from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=4.0, description="CGPA must be between 0.0 and 4.0")

new_student = {'name': "Dayan Atif", 'age': '21', 'email': 'dayan@gmail.com', 'cgpa': '3.96'}

student = Student(**new_student)

print(student)