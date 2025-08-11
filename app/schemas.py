from pydantic import BaseModel


class StudentBase(BaseModel):
    name: str
    number: str
    # photo: bytes


class StudentCreate(StudentBase):
    password: str


class StudentOut(StudentBase):
    ...