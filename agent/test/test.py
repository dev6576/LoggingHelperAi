
import faiss
import pickle

index = faiss.read_index(r"D:\GitHub\LogManager\agent\vector_store\index.faiss")
print("Number of vectors in index:", index.ntotal)
print("Vector count:", index.ntotal)
print("Vector dimension:", index.d)


with open(r"D:\GitHub\LogManager\agent\vector_store\metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
    print("Metadata contents:", metadata)