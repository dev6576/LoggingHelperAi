from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class LogRouter:
    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        :param index_path: Path to FAISS index file
        :param metadata_path: Path to corresponding metadata (e.g., code snippets)
        :param model_name: Sentence transformer model for embeddings
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logging.error("FAISS index or metadata not found. Please run ingestion first.")
            self.index = None
            self.metadata = []
            return
        model_name = 'microsoft/codebert-base'
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        logging.info(f"Loaded FAISS index from {index_path}")

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logging.info(f"Loaded metadata from {metadata_path} with {len(self.metadata)} entries")

    def route(self, log_text: str, top_k: int = 5):
        """
        Given a log string, find the top-k relevant code snippets/components.

        :param log_text: Raw log string
        :param top_k: Number of top code chunks to return
        :return: List of dicts with matched code info
        """
        logging.info(f"Routing log text of length {len(log_text)}")
        log_embedding = self.model.encode([log_text])
        logging.debug(f"Generated log embedding: {log_embedding[0][:5]}...")

        distances, indices = self.index.search(np.array(log_embedding).astype('float32'), top_k)
        logging.info(f"FAISS search returned indices: {indices[0]} with distances: {distances[0]}")

        results = []
        for idx, score in zip(indices[0], distances[0]):
            match = self.metadata[idx]
            match['score'] = float(score)
            logging.info(f"Matched component: {match.get('component', 'unknown')} with score: {score}")
            results.append(match)
        logging.debug(f"Results: {results[:top_k]}")
        logging.info(f"Routing complete. Top {top_k} results returned.")
        return results