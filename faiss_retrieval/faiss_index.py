import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss

class FaissRetriever:
    def __init__(self, index_path="models/two_tower/faiss_index.bin", mappings_path="models/two_tower/two_tower_mappings.pkl"):
        self.index_path = index_path
        self.mappings_path = mappings_path
        self.index = None
        self.mappings = None
        self.book_embeddings = None
        
    def build_index(self, book_embeddings_path="models/two_tower/book_embeddings.npy"):
        print("Building FAISS index for candidate retrieval...")
        if not os.path.exists(book_embeddings_path):
            raise FileNotFoundError("Book embeddings file not found. Run two-tower training first.")
            
        self.book_embeddings = np.load(book_embeddings_path)
        
        # Load mappings to resolve names
        with open(self.mappings_path, "rb") as f:
            self.mappings = pickle.load(f)
            
        dimension = self.book_embeddings.shape[1]
        
        # We use IndexFlatIP (Inner Product) since embeddings are L2 normalized
        # This makes dot product equivalent to cosine similarity!
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add book embeddings (row 0 is padding, we add it but it won't be queried or we can handle it)
        # To avoid query matching padding index, we can just add all of them, or keep them.
        self.index.add(self.book_embeddings)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index built and saved to {self.index_path}. Total vectors indexed: {self.index.ntotal}")
        
    def load_index(self):
        print("Loading FAISS index and mappings...")
        if not (os.path.exists(self.index_path) and os.path.exists(self.mappings_path)):
            raise FileNotFoundError("FAISS index or mappings not found. Build them first.")
            
        self.index = faiss.read_index(self.index_path)
        with open(self.mappings_path, "rb") as f:
            self.mappings = pickle.load(f)
        self.book_embeddings = np.load("models/two_tower/book_embeddings.npy")
        print("FAISS index loaded successfully.")
        
    def retrieve_candidates(self, user_embedding, top_k=100):
        if self.index is None:
            raise ValueError("Index not loaded.")
            
        # Ensure embedding shape is [1, 128]
        user_emb = np.array(user_embedding, dtype=np.float32).reshape(1, -1)
        
        # Query FAISS index
        # D: distances (cosine similarities), I: indices
        D, I = self.index.search(user_emb, top_k)
        
        candidate_ids = []
        scores = []
        
        book_encoder = self.mappings["book_encoder"]
        
        for sim, idx in zip(D[0], I[0]):
            # Skip padding index (0)
            if idx == 0:
                continue
                
            # Convert internal model code back to original book ID
            book_idx = idx - 1
            if book_idx < len(book_encoder.classes_):
                orig_book_id = book_encoder.classes_[book_idx]
                candidate_ids.append(int(orig_book_id))
                scores.append(float(sim))
                
        return candidate_ids, scores

def run_faiss_benchmark():
    print("\n=== FAISS RETRIEVAL LATENCY BENCHMARK ===")
    
    # Check if files exist
    if not os.path.exists("models/two_tower/book_embeddings.npy"):
        print("Embeddings missing. Skipping benchmark. Complete Two-Tower training first.")
        return
        
    # Set up retriever
    retriever = FaissRetriever()
    retriever.build_index()
    
    user_embeddings = np.load("models/two_tower/user_embeddings.npy")
    book_embeddings = np.load("models/two_tower/book_embeddings.npy")
    
    # Select a random user embedding (excluding index 0 padding)
    test_user_idx = np.random.randint(1, len(user_embeddings))
    user_emb = user_embeddings[test_user_idx].reshape(1, -1)
    
    # 1. FAISS Search Benchmark
    num_queries = 1000
    t0 = time.time()
    for _ in range(num_queries):
        D_faiss, I_faiss = retriever.index.search(user_emb, 100)
    faiss_time = (time.time() - t0) / num_queries * 1000 # in ms
    
    # 2. Brute-Force Numpy Search Benchmark
    t0 = time.time()
    for _ in range(num_queries):
        # Calculate dot products
        scores = np.dot(user_emb, book_embeddings.T).flatten()
        # Get top 100 indices
        top_100_idx = np.argpartition(scores, -100)[-100:]
        top_100_sorted = top_100_idx[np.argsort(scores[top_100_idx])][::-1]
    numpy_time = (time.time() - t0) / num_queries * 1000 # in ms
    
    print(f"Num queries tested: {num_queries}")
    print(f"FAISS Search Average Latency:      {faiss_time:.4f} ms")
    print(f"NumPy Brute-force Average Latency: {numpy_time:.4f} ms")
    print(f"FAISS speedup:                     {numpy_time / max(1e-9, faiss_time):.1f}x")
    
    # Validate correctness
    faiss_results = set(I_faiss[0][:10])
    numpy_results = set(top_100_sorted[:10])
    overlap = len(faiss_results.intersection(numpy_results))
    print(f"Accuracy overlap on top-10 candidates: {overlap * 10}%")
    
    # Save benchmark report
    os.makedirs("reports", exist_ok=True)
    with open("reports/faiss_benchmark.txt", "w") as f:
        f.write("=== FAISS candidate retrieval benchmark ===\n")
        f.write(f"FAISS Search Latency:      {faiss_time:.4f} ms\n")
        f.write(f"NumPy Brute-Force Latency: {numpy_time:.4f} ms\n")
        f.write(f"Speedup Factor:            {numpy_time / max(1e-9, faiss_time):.2f}x\n")
        f.write(f"Top-10 Overlap:            {overlap * 10}%\n")
    print("Benchmark report written to reports/faiss_benchmark.txt")

if __name__ == "__main__":
    run_faiss_benchmark()
