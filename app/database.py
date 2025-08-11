from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 数据库访问地址
SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://root:123456@localhost:3306/test'

# 启动引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

# 启动会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 数据模型基类
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
