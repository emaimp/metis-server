from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import init_db
from app.routers.chat import router as chat_router
from app.routers.chats import router as chats_router
from app.routers.models import router as models_router
from app.routers.voices import router as voices_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init SQLite DB for chat persistence
    init_db()
    yield


app = FastAPI(title="Metis Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(chats_router)
app.include_router(models_router)
app.include_router(voices_router)
