from fastapi import FastAPI
from app.database import Base, engine
from app.routes import router


# 创建数据库表
Base.metadata.create_all(bind=engine)
app = FastAPI()

app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True)