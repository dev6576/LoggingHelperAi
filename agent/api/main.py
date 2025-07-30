from fastapi import FastAPI, UploadFile, File, HTTPException
from core.log_router import LogRouter
from core.llm_agent import LLMComponentAgent
from core.github_ingestor import GitHubIngestor
import os
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

# Paths for vector index and metadata
INDEX_PATH = "vector_store/index.faiss"
METADATA_PATH = "vector_store/metadata.pkl"

# Initialize core services
log_router = LogRouter(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
llm_agent = LLMComponentAgent()

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server on 127.0.0.1:8080")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8080, reload=True)

@app.get("/")
def root():
    logging.info("Root endpoint accessed")
    return {"status": "Debugging agent API running ✅"}

@app.post("/ingest/")
def ingest_repo(repo_url: str):
    logging.info(f"Received ingest request for repo: {repo_url}")
    try:
        ingestor = GitHubIngestor()  # Optionally pass GitHub token here
        components = ingestor.ingest_and_store(repo_url, INDEX_PATH, METADATA_PATH)
        logging.info(f"Successfully ingested repo: {repo_url} with components: {components}")
        return {"status": "success", "components": components}
    except Exception as e:
        logging.error(f"Error ingesting repo {repo_url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-log/")
async def upload_log(file: UploadFile = File(...)):
    logging.info(f"Received log upload: {file.filename}")
    # Save uploaded file temporarily
    temp_dir = "temp_logs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logging.info(f"Saved uploaded log to {file_path}")

    log_text = open(file_path, "r", encoding="utf-8").read()
    logging.info(f"Log file content: {log_text[:200]}...")  # Show first 200 chars

    # Route log to component
    component = log_router.route(log_text)
    logging.info(f"Mapped log to component: {component['']}")

    # Analyze using LLM
    analysis = llm_agent.analyze(log_text, component)
    logging.info(f"LLM analysis complete for component: {component}")

    return {
        "mapped_component": component,
        "analysis": analysis
    }