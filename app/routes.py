from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response, status, Request, Form
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import jwt
import os
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import select
from .schemas import StudentCreate, StudentOut
from .database import get_db
from .models import Student
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from deepface import DeepFace
import numpy as np
import cv2
import shutil

# 加密上下文，用于密码哈希和验证
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# 用于验证和生成 JWT 令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# 密钥和令牌过期时间设置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 人脸识别配置
IMAGE_DIR = "uploads"
os.makedirs(IMAGE_DIR, exist_ok=True)
RECOGNITION_MODEL = "VGG-Face"  # 可配置的识别模型
DISTANCE_METRIC = "cosine"  # 可配置的距离度量
RECOGNITION_THRESHOLD = 0.4  # 识别阈值，越小越严格

router = APIRouter()


# 工具函数：验证人脸相似度
def verify_faces(img1_path: str, img2_path: str):
    """比较两张图片中的人脸相似度"""
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name=RECOGNITION_MODEL, distance_metric=DISTANCE_METRIC)
        return result
    except Exception as e:
        print(f"人脸验证失败: {e}")
        return {"verified": False, "distance": float('inf')}


# 精簡後的人臉識別路由
@router.post("/recognize")
async def recognize_face(
        photo: UploadFile = File(...),
        class_code: str = Form(...),
        db: Session = Depends(get_db)
):
    try:
        if not class_code:
            raise HTTPException(status_code=400, detail="班级代碼不能为空")

        # 保存上传的考勤照片
        temp_photo_path = os.path.join(IMAGE_DIR, f"temp_{photo.filename}")
        with open(temp_photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        # 在指定班级中查找所有学生
        students_in_class = db.scalars(select(Student).where(Student.class_code == class_code)).all()

        if not students_in_class:
            os.remove(temp_photo_path)
            return {"recognized": False, "message": "该班级没有学生数据，无法进行考勤。"}

        # 检查上传的照片中是否有人脸
        face_detected = True
        try:
            detected_faces = DeepFace.extract_faces(temp_photo_path, detector_backend='opencv', enforce_detection=True)
            if not detected_faces:
                face_detected = False
        except Exception:
            face_detected = False

        if not face_detected:
            os.remove(temp_photo_path)
            return {"recognized": False, "message": "未在照片中检测到人脸，考勤失败。"}

        # 对比上传照片与班级中每个学生的照片
        best_match = None
        best_distance = float('inf')

        for student in students_in_class:
            if student.photo and os.path.exists(os.path.join(IMAGE_DIR, student.photo)):
                result = verify_faces(temp_photo_path, os.path.join(IMAGE_DIR, student.photo))
                if result["verified"] and result["distance"] < best_distance:
                    best_distance = result["distance"]
                    best_match = student

        # 计算置信度并返回结果
        confidence = 1.0 - min(best_distance / RECOGNITION_THRESHOLD, 1.0) if best_match else 0.0
        recognized = best_match is not None and best_distance < RECOGNITION_THRESHOLD

        response_data = {
            "recognized": recognized,
            "face_detected": face_detected,
            "confidence": confidence,
            "threshold": RECOGNITION_THRESHOLD,
            "model": RECOGNITION_MODEL
        }

        if recognized:
            response_data.update({
                "student_id": best_match.id,
                "student_number": best_match.number,
                "student_name": best_match.name,
                "message": "人脸识别成功"  # 成功时返回简短的成功訊息
            })
        else:
            response_data.update({
                "message": "人脸不匹配，考勤失败"
            })

        return response_data

    except HTTPException as e:
        return {"recognized": False, "message": e.detail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人脸识别处理错误: {str(e)}")
    finally:
        # 清理临时文件
        try:
            if 'temp_photo_path' in locals() and os.path.exists(temp_photo_path):
                os.remove(temp_photo_path)
        except Exception as e:
            print(f"清理临时文件失败: {e}")


@router.post("/register")
async def register_student(
        name: str = Form(...),
        student_number: str = Form(...),
        password: str = Form(...),
        role: str = Form(...),
        class_code: Optional[str] = Form(None),
        photo: Optional[UploadFile] = File(None),
        db: Session = Depends(get_db)
):
    """
    注册新用户 (学生或教师)
    """
    if role == "student":
        if not photo or not class_code:
            raise HTTPException(status_code=400, detail="学生注册需要照片和班级代碼。")

        existing_student = db.scalar(select(Student).where(Student.number == student_number))
        if existing_student:
            raise HTTPException(status_code=409, detail="学号已存在。")

        photo_filename = f"{student_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        photo_path = os.path.join(IMAGE_DIR, photo_filename)
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        db_student = Student(
            number=student_number,
            name=name,
            password=pwd_context.hash(password),
            photo=photo_filename,
            class_code=class_code
        )

        db.add(db_student)
        db.commit()
        db.refresh(db_student)
        return {"success": True, "message": "註冊成功！"}

    elif role == "teacher":
        return {"success": True, "message": "教師註冊成功！"}

    else:
        raise HTTPException(status_code=400, detail="無效的角色。")


@router.post("/login")
async def login_for_access_token(
        request: Request,
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    用户登录并返回 JWT token
    """
    student = db.scalar(select(Student).where(Student.number == form_data.username))
    if not student or not pwd_context.verify(form_data.password, student.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = jwt.encode(
        {"sub": student.number, "id": student.id, "exp": datetime.utcnow() + access_token_expires},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "message": "登入成功",
        "token": access_token,
        "role": "student",
        "id": student.id,
        "class_code": student.class_code,
        "student_name": student.name
    }