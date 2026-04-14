"""FastAPI application — entry point for ``uvicorn prompt_similarity.app:app``.

Registers all route modules and manages the application lifespan (OpenAI
client init, database schema creation, vector cache warm-up).
"""

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI

from prompt_similarity import cache
from prompt_similarity.db import init_db
from prompt_similarity.app_state import init_client
from prompt_similarity.routes import embeddings, search, analysis, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create OpenAI client, ensure DB schema, warm the cache."""
    init_client()
    init_db()
    cache.rebuild()
    yield


app = FastAPI(
    title="Prompt Similarity Service",
    version="0.3.0",
    lifespan=lifespan,
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(embeddings.router)
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(health.router)
