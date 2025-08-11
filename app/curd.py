from sqlalchemy.orm import Session
from . import models, schemas

def get_students(db: Session, student_number: str):
    return db.query(models.Student).filter(models.Student.number == student_number)