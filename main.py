# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routes import router

Base.metadata.create_all(bind=engine)
app = FastAPI()

# 正確配置 CORS 中間件
# 確保你的 Android 應用程式能夠與後端通信
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在開發階段，可以允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', reload=True, host='0.0.0.0')