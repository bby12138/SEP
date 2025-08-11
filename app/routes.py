from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response, status, Request
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


# 工具函数：提取人脸特征
def extract_face_features(image_path: str) -> list[dict[str, Any]] | None:
    try:
        # 使用DeepFace提取特征
        face_obj = DeepFace.extract_faces(
            img_path=image_path,
            target_size=(224, 224),
            detector_backend="opencv"
        )

        if not face_obj:
            return None

        # 返回第一个检测到的人脸的特征向量
        return DeepFace.represent(
            img_path=face_obj[0]["face"],
            model_name=RECOGNITION_MODEL,
            enforce_detection=False
        )
    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        return None


# 工具函数：验证人脸相似度
def verify_faces(img1_path: str, img2_path: str) -> Dict:
    try:
        # 使用DeepFace验证两张人脸
        return DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=RECOGNITION_MODEL,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False
        )
    except Exception as e:
        print(f"人脸验证错误: {str(e)}")
        return {"verified": False, "distance": float('inf')}


# 工具函数：检测并裁剪人脸
def detect_and_crop_face(image_path: str, output_path: str) -> bool:
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return False

        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用OpenCV的人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return False

        # 获取第一个检测到的人脸
        (x, y, w, h) = faces[0]

        # 裁剪人脸区域并保存
        face_roi = img[y:y + h, x:x + w]
        cv2.imwrite(output_path, face_roi)

        return True
    except Exception as e:
        print(f"人脸检测和裁剪错误: {str(e)}")
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_student_by_number(db: Session, number: str):
    query = select(Student).where(Student.number == number)
    return db.execute(query).scalars().first()


def authenticate_student(db: Session, number: str, password: str):
    student = get_student_by_number(db, number)
    if not student:
        return False
    if not verify_password(password, student.password):
        return False
    return student


async def get_current_student(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        number: str = payload.get("sub")
        if number is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    student = get_student_by_number(db, number)
    if student is None:
        raise credentials_exception
    return student


@router.post('/create_or_register')
async def create_or_register(student: StudentCreate, db_session: Session = Depends(get_db)):
    query = select(Student).where(Student.number == student.number)
    records = db_session.execute(query).scalars().all()
    if records:
        raise HTTPException(status_code=400, detail=f'student  {student.number} already exists')
    hashed_password = get_password_hash(student.password)
    new_student = Student(name=student.name, number=student.number, password=hashed_password)
    db_session.add(new_student)
    db_session.commit()
    db_session.refresh(new_student)
    return {"message": "Student created/registered successfully"}


@router.post('/login')
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    student = authenticate_student(db, form_data.username, form_data.password)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": student.number}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post('/photo/')
async def upload_photos(
        file: UploadFile = File(...),
        db_session: Session = Depends(get_db),
        current_student: Student = Depends(get_current_student)
):
    # 1. 查询学生是否存在
    student = current_student

    # 2. 保存文件到服务器
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"student_{student.id}_{timestamp}{file_ext}"
    file_path = os.path.join(upload_dir, unique_filename)

    # 保存原始文件
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 3. 检测并裁剪人脸（可选步骤）
    face_cropped = False
    face_path = None
    if file_ext in ['.jpg', '.jpeg', '.png']:
        face_filename = f"face_{student.id}_{timestamp}{file_ext}"
        face_path = os.path.join(upload_dir, face_filename)
        face_cropped = detect_and_crop_face(file_path, face_path)

    try:
        # 4. 更新学生照片路径
        student.photo = face_path if face_cropped else file_path
        db_session.commit()
        db_session.refresh(student)
    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Photo updated successfully", "face_detected": face_cropped}


@router.get('/getstudent', response_model=StudentOut)
async def get_student(
        current_student: Student = Depends(get_current_student)
):
    return current_student


@router.get('/student/photo')
async def get_student_photo(
        current_student: Student = Depends(get_current_student)
):
    if not current_student.photo:
        raise HTTPException(status_code=404, detail='Photo not found')

    try:
        # 检查文件是否存在
        if not os.path.isfile(current_student.photo):
            raise FileNotFoundError

        # 获取文件扩展名以确定媒体类型
        file_extension = os.path.splitext(current_student.photo)[1].lower()
        if file_extension == '.jpg' or file_extension == '.jpeg':
            media_type = 'image/jpeg'
        elif file_extension == '.png':
            media_type = 'image/png'
        elif file_extension == '.gif':
            media_type = 'image/gif'
        else:
            raise HTTPException(status_code=400, detail='Unsupported image type')

        with open(current_student.photo, "rb") as file:
            photo_data = file.read()
        return Response(
            content=photo_data,
            media_type=media_type
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Photo not found')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 优化后的人脸识别接口
@router.post("/api/recognize")
async def recognize(face_image: UploadFile = File(...)):
    # 创建临时目录
    temp_dir = os.path.join(IMAGE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 保存上传的图片
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_ext = os.path.splitext(face_image.filename)[1].lower()

    # 确保文件扩展名是有效的图片格式
    if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return {"recognized": False, "message": "不支持的文件格式，仅支持JPG、PNG、BMP格式"}

    temp_image_filename = f"query_{timestamp}{file_ext}"
    temp_image_path = os.path.join(temp_dir, temp_image_filename)

    # 保存上传的图片
    with open(temp_image_path, "wb") as f:
        contents = await face_image.read()
        f.write(contents)

    # 人脸检测和裁剪
    face_detected = False
    face_path = os.path.join(temp_dir, f"query_face_{timestamp}{file_ext}")

    if detect_and_crop_face(temp_image_path, face_path):
        query_image = face_path
        face_detected = True
    else:
        # 如果无法检测到人脸，使用原始图像
        query_image = temp_image_path

    try:
        db = next(get_db())

        # 获取所有已注册且有照片的学生
        students = db.query(Student).filter(Student.photo != None).all()

        best_match = None
        best_distance = float('inf')

        # 遍历所有学生进行比对
        for student in students:
            if not student.photo or not os.path.isfile(student.photo):
                continue

            # 验证人脸相似度
            result = verify_faces(query_image, student.photo)

            if result["verified"] and result["distance"] < best_distance:
                best_distance = result["distance"]
                best_match = student

        # 计算置信度 (距离越小，置信度越高)
        confidence = 1.0 - min(best_distance / RECOGNITION_THRESHOLD, 1.0) if best_match else 0.0

        # 判断是否识别成功
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
                "student_name": best_match.name
            })
        else:
            response_data.update({
                "message": "未识别到匹配的学生"
            })

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人脸识别处理错误: {str(e)}")
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(face_path):
                os.remove(face_path)
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")