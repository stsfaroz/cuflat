import numpy as np
import time
from cuflat import GPUVectorStore, normalize_embeddings

def create_test_data(num_docs=10000, embed_dim=1024):
    """Create normalized embeddings and a query vector"""
    # Generate random embeddings
    random_embeddings = np.random.rand(num_docs, embed_dim).astype(np.float32) * 2 - 1
    embeddings = normalize_embeddings(random_embeddings)
    
    # Use one of the embeddings as query for testing
    query = embeddings[100].copy()
    
    return embeddings, query

def main():
    print("Creating test data...")
    embeddings, query = create_test_data(num_docs=3000, embed_dim=384)
    print(f"Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Initialize GPU vector store
    store = GPUVectorStore()
    
    try:
        print("Adding embeddings to GPU index...")
        store.add(embeddings)
        
        # Warmup searches
        warmup_runs = 5
        print(f"Running {warmup_runs} warmup searches...")
        for _ in range(warmup_runs):
            start_time = time.perf_counter()
            store.search(query, 5)
            end_time = time.perf_counter()
            search_time_ms = (end_time - start_time) * 1000
        
            # Print results
            print(f"\nSearch completed in {search_time_ms:.2f} ms")
            
        # Perform search and measure time
        k = 5
        start_time = time.perf_counter()
        similarities, indices = store.search(query, k)
        end_time = time.perf_counter()
        
        search_time_ms = (end_time - start_time) * 1000
        
        # Print results
        print(f"\nSearch completed in {search_time_ms:.2f} ms")
        print(f"Top {k} results:")
        for i in range(k):
            print(f"  Rank {i+1}: Index {indices[0][i]:5d}, Similarity {similarities[0][i]:.6f}")
    
    finally:
        # Always cleanup GPU resources
        store.cleanup()

if __name__ == "__main__":
    main()