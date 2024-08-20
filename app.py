import io
import logging
import os
import threading
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from sklearn.decomposition import PCA

from tagmatch.fuzzysearcher import FuzzyMatcher
from tagmatch.logging_config import setup_logging
from tagmatch.vec_db import Embedder, EmbedReduce, PCAEmbedReduce, VecDB

if not os.path.exists("./data"):
    os.makedirs("./data")

setup_logging(file_path="./data/service.log")


class Settings(BaseSettings):
    model_name: str
    cache_dir: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    reduced_embed_dim: int

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI()
logger = logging.getLogger("fastapi")
# In-memory storage for tags
app.names_storage = []
app.tag_synonyms = defaultdict(set)

# Placeholder for the semantic search components
embedder = Embedder(model_name=settings.model_name, cache_dir=settings.cache_dir)

embed = EmbedReduce(embedder=embedder)
pca = PCAEmbedReduce(embedder=embedder, pca_pkl_path="pca.pkl")

vec_db = VecDB(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    collection=settings.qdrant_collection,
    vector_size=settings.reduced_embed_dim,
)

app.fuzzy_matcher = FuzzyMatcher([])

# Flag to track background task status
task_running = threading.Event()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")

    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        logger.info(f"Request Body: {body.decode('utf-8')}")

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.post("/upload-csv/")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if task_running.is_set():
        raise HTTPException(
            status_code=400, detail="A task is already running. Please try again later."
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file (needs to env with '.csv'.",
        )

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), sep=None, header=0)

    if "name" not in df.columns:
        raise HTTPException(
            status_code=400, detail="CSV file must contain a 'name' column."
        )

    names_storage = df["name"].dropna().unique().tolist()

    if len(names_storage) == 0:
        raise HTTPException(status_code=400, detail="No names found in the CSV file.")

    # Return error if the collection is already existing (and it's populated)
    if vec_db.collection_exists():
        raise HTTPException(
            status_code=400,
            detail="Collection already exists. Please delete the collection first.",
        )
    else:
        vec_db._create_collection()

    # Set the flag to indicate that a task is running
    task_running.set()

    # Add background task to process the CSV file
    background_tasks.add_task(process_csv, names_storage)

    return {
        "message": "File accepted for processing. Names will be extracted and stored in the background."
    }


def process_csv(names_storage: List[str]):
    try:
        # Store embedded vectors for semantic search
        for name in names_storage:
            vector = pca(name)
            vec_db.store(vector, {"name": name})

        app.names_storage = names_storage
        app.fuzzy_matcher = FuzzyMatcher(app.names_storage)
    finally:
        # Clear the flag to indicate that the task has completed
        task_running.clear()


@app.delete("/purge/")
async def delete_collection():
    vec_db.remove_collection()
    app.names_storage = []
    return {"message": "DB deleted successfully."}


@app.get("/search/")
async def search(query: str, k: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    # Fuzzy search
    fuzzy_matches = app.fuzzy_matcher.get_top_k_matches(query, k)
    # Semantic search

    query_vector = pca(query)

    semantic_matches = vec_db.find_closest(query_vector, k)
    semantic_matches: List[Tuple[str, float]] = [
        (m.payload["name"], m.score) for m in semantic_matches
    ]

    tags_has_synonym_lists: List[List[str]] = [
        app.tag_synonyms[m[0]] for m in semantic_matches if m[0] in app.tag_synonyms
    ]

    semantic_matches = filter(lambda m: m[0] not in app.tag_synonyms, semantic_matches)

    tags_has_synonym: List[str] = []

    for tag_list in tags_has_synonym_lists:
        tags_has_synonym += tag_list

    query_vecs_has_synonyms = [pca(query) for query in tags_has_synonym]

    syn_sem_matches_lists = [
        vec_db.find_closest(tag, k) for tag in query_vecs_has_synonyms
    ]

    syn_sem_matches = []
    for matches in syn_sem_matches_lists:
        syn_sem_matches += matches

    syn_sem_matches = [
        (match.payload["name"], match.score) for match in syn_sem_matches
    ]

    syn_sem_matches += semantic_matches
    syn_sem_matches.sort(key=lambda t: t[1], reverse=True)

    result = []
    seen = set()

    for tpl in syn_sem_matches:
        if tpl[0] not in seen:
            result.append(tpl)
            seen.add(tpl[0])

    # Formatting the response
    semantic_results = [{"name": match[0], "score": match[1]} for match in result[:k]]

    typo_results = [
        {"name": match["matched"], "score": match["score"]} for match in fuzzy_matches
    ]

    response = {"match": {"semantic": semantic_results, "typo": typo_results}}

    return JSONResponse(content=response)


@app.get("/task-status/")
async def task_status():
    if task_running.is_set():
        return {"status": "running", "processed_items": vec_db.get_item_count()}
    else:
        return {"status": "finished", "nb_items_stored": vec_db.get_item_count()}


@app.post("/add_synonym/")
async def update_synonym(tag_1: str, tag_2: str):

    if tag_1 != tag_2:
        app.tag_synonyms[tag_1].update([tag_2])

    return {tag_1: list(app.tag_synonyms[tag_1])}


@app.get("/get_tag_list/")
async def get_tag_list():
    return {"tag_list": app.names_storage}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
