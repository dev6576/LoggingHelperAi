# core/vector_store.py

import os
import pickle
import faiss
import numpy as np
import logging
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class VectorStoreBuilder:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Use local code-aware model
        model_name = 'microsoft/codebert-base'
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logging.info(f"Using local embedding model: {model_name} (dimension={self.dimension})")
        logging.info(f"Initialized VectorStoreBuilder with index_path={index_path} and metadata_path={metadata_path}")


    def build_store(self, component_sources: dict, batch_size: int = 10):
            logging.info(f"Building vector store for {len(component_sources)} components")
            index = faiss.IndexFlatL2(self.dimension)
            metadata = []

            for component, files_dict in component_sources.items():
                for file_path, code_text in files_dict.items():
                    try:
                        embedding = self.model.encode([code_text], show_progress_bar=False)[0]
                        index.add(np.array([embedding], dtype='float32'))
                        metadata.append({
                            "component": component,
                            "file": file_path,
                            "code": code_text
                        })
                        logging.debug(f"Indexed component: {component}, file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error embedding file {file_path} in component {component}: {e}")
                        time.sleep(2)
                        continue

            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(index, self.index_path)
            logging.info(f"FAISS index saved to {self.index_path}")

            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            logging.info(f"Metadata saved to {self.metadata_path}")


