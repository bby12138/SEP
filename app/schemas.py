from pydantic import BaseModel


class StudentBase(BaseModel):
    name: str
    number: str
    # photo: bytes


class StudentCreate(StudentBase):
    password: str
    # 新增: 注册时需要的班级代碼
    class_code: str


class StudentOut(StudentBase):
    pass