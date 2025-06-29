import numpy as np
import ctypes
import os
from typing import Tuple, Optional

class GPUVectorStore:
    """
    FAISS-like GPU vector store for cosine similarity search
    """
    
    def __init__(self, lib_path: str = './cuflat/libsearch.so'):
        """
        Initialize GPU Vector Store
        
        Args:
            lib_path: Path to the compiled CUDA shared library
        """
        self.lib_path = lib_path
        self.lib = None
        self.is_initialized = False
        self.num_docs = 0
        self.embed_dim = 0
        self._load_library()
    
    def _load_library(self):
        """Load the CUDA shared library and define function signatures"""
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Shared library '{self.lib_path}' not found.")
        
        try:
            self.lib = ctypes.CDLL(self.lib_path)
        except OSError as e:
            raise OSError(f"Error loading shared library: {e}")
        
        # Define function signatures
        self.lib.init_gpu_search.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        self.lib.init_gpu_search.restype = ctypes.c_int
        
        self.lib.gpu_search_topk.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        self.lib.gpu_search_topk.restype = None
        
        self.lib.cleanup_gpu_search.argtypes = []
        self.lib.cleanup_gpu_search.restype = None
    
    def add(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the GPU index (FAISS-like interface)
        
        Args:
            embeddings: 2D numpy array of shape (num_docs, embed_dim)
                       Should be normalized for cosine similarity
        """
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        self.num_docs, self.embed_dim = embeddings.shape
        
        # Ensure embeddings are float32 and contiguous
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        if not embeddings.flags['C_CONTIGUOUS']:
            embeddings = np.ascontiguousarray(embeddings)
        
        # Initialize GPU resources
        c_embeddings_ptr = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret_code = self.lib.init_gpu_search(c_embeddings_ptr, self.num_docs, self.embed_dim)
        
        if ret_code != 0:
            raise RuntimeError(f"GPU initialization failed with code: {ret_code}")
        
        self.is_initialized = True
        print(f"GPU index initialized: {self.num_docs} docs, {self.embed_dim} dims")
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k most similar vectors (FAISS-like interface)
        
        Args:
            query: Query vector of shape (embed_dim,) or (1, embed_dim)
            k: Number of top results to return
        
        Returns:
            Tuple of (similarities, indices) both of shape (1, k) to match FAISS format
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized. Call add() first.")
        
        # Handle both 1D and 2D query inputs
        if len(query.shape) == 2:
            if query.shape[0] != 1:
                raise ValueError("Only single query supported. Query shape should be (1, embed_dim)")
            query = query[0]  # Convert to 1D
        elif len(query.shape) != 1:
            raise ValueError("Query must be 1D or 2D array")
        
        if query.shape[0] != self.embed_dim:
            raise ValueError(f"Query dimension {query.shape[0]} doesn't match index dimension {self.embed_dim}")
        
        # Ensure query is float32 and contiguous
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        if not query.flags['C_CONTIGUOUS']:
            query = np.ascontiguousarray(query)
        
        # Prepare output arrays
        output_indices = np.zeros(k, dtype=np.int32)
        output_similarities = np.zeros(k, dtype=np.float32)
        
        # Get C pointers
        c_query_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_output_indices_ptr = output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_output_similarities_ptr = output_similarities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Perform search
        self.lib.gpu_search_topk(c_query_ptr, c_output_indices_ptr, c_output_similarities_ptr, k)
        
        # Return in FAISS format (2D arrays)
        return output_similarities.reshape(1, -1), output_indices.reshape(1, -1)
    
    def cleanup(self) -> None:
        """Clean up GPU resources"""
        if self.is_initialized:
            self.lib.cleanup_gpu_search()
            self.is_initialized = False
            print("GPU resources cleaned up.")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()



def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  
    return embeddings / norms
