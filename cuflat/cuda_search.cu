#include <cuda_runtime.h>
#include <cublas_v2.h> 
#include <cub/cub.cuh>        
#include <thrust/device_ptr.h>
#include <thrust/sequence.h> 
#include <thrust/execution_policy.h> 
#include <algorithm>           
#include <stdio.h>            
#include <cfloat>         

// Macro for checking CUDA runtime API errors
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Macro for checking cuBLAS API errors
#define cublasCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"CUBLAS Error (Status %d): %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}


// External C linkage for Python ctypes wrapper
extern "C" {
    int init_gpu_search(const float* embeddings, int num_docs, int embed_dim);
    void gpu_search_topk(const float* query, int* indices, float* similarities, int top_k);
    void cleanup_gpu_search();
}


// Global state variables for GPU resources
static float* g_d_embeddings = nullptr;        
static float* g_d_query = nullptr;            
static float* g_d_similarities = nullptr;      
static int* g_d_original_indices = nullptr;   

// For CUB sorting, these will store the fully sorted values and their corresponding indices
static float* g_d_cub_sorted_values = nullptr;
static int* g_d_cub_sorted_indices = nullptr;

static void* g_d_cub_temp_storage = nullptr;   
static size_t g_cub_temp_storage_bytes = 0;   

// Global static pointers for custom top-k kernel's output to avoid re-allocating
static float* g_d_temp_top_values = nullptr;
static int* g_d_temp_top_indices = nullptr;
const int MAX_SUPPORTED_TOP_K_CUSTOM = 8;

static cublasHandle_t g_cublas_handle = nullptr; 
static cudaStream_t g_stream = nullptr;        

static int g_num_docs = 0;                    
static int g_embed_dim = 0;                    
static bool g_initialized = false;            


/**
 * @brief Optimized CUDA kernel for computing dot products for a fixed dimension of 384.
 * This version uses a tiled approach for better memory coalescing and warp-level accumulation.
 * The shared memory size for the query is hardcoded to 384.
 *
 * @tparam BLOCK_DIM_X Number of threads in X dimension (typically 32 for warps).
 * @tparam BLOCK_DIM_Y Number of threads in Y dimension (number of documents processed per block).
 * @param embeddings Device pointer to the matrix of document embeddings.
 * @param query Device pointer to the query vector.
 * @param similarities Device pointer to output array for similarity scores.
 * @param num_docs Total number of documents.
 * @param embed_dim Dimension of embeddings (should be 384 for this specialized kernel).
 */
template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 2)
gemv_optimized_384( // Renamed
    const float* __restrict__ embeddings,
    const float* __restrict__ query,
    float* __restrict__ similarities,
    int num_docs,
    int embed_dim) {

    const int tid_x = threadIdx.x; 
    const int tid_y = threadIdx.y;
    const int block_width = BLOCK_DIM_X; 
    const int block_height = BLOCK_DIM_Y; 
    const int threads_per_block = blockDim.x * blockDim.y;

    // Shared memory for query vector - FIXED TO 384
    __shared__ float s_query[384];

    // Cooperatively load query into shared memory.
    for (int i = tid_x + tid_y * blockDim.x; i < embed_dim; i += threads_per_block) {
        s_query[i] = __ldg(&query[i]);
    }
    __syncthreads(); 

    const int doc_id_base = blockIdx.x * block_height + tid_y;

    if (doc_id_base < num_docs) {
        float sum = 0.0f;
        const float* doc_ptr = &embeddings[doc_id_base * embed_dim];

        #pragma unroll
        for (int i = tid_x; i < embed_dim; i += block_width) {
            sum += __ldg(&doc_ptr[i]) * s_query[i];
        }

        // Warp-level reduction for the partial sums.
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid_x == 0) {
            similarities[doc_id_base] = sum;
        }
    }
}

/**
 * @brief Optimized CUDA kernel for computing dot products for a fixed dimension of 512.
 * This version mirrors the 384-dim optimized kernel with shared memory fixed to 512.
 *
 * @tparam BLOCK_DIM_X Number of threads in X dimension (typically 32 for warps).
 * @tparam BLOCK_DIM_Y Number of threads in Y dimension (number of documents processed per block).
 * @param embeddings Device pointer to the matrix of document embeddings.
 * @param query Device pointer to the query vector.
 * @param similarities Device pointer to output array for similarity scores.
 * @param num_docs Total number of documents.
 * @param embed_dim Dimension of embeddings (should be 512 for this specialized kernel).
 */
template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 2)
gemv_optimized_512( // Renamed
    const float* __restrict__ embeddings,
    const float* __restrict__ query,
    float* __restrict__ similarities,
    int num_docs,
    int embed_dim) {

    const int tid_x = threadIdx.x; 
    const int tid_y = threadIdx.y; 
    const int block_width = BLOCK_DIM_X; 
    const int block_height = BLOCK_DIM_Y; 
    const int threads_per_block = blockDim.x * blockDim.y; 

    // Shared memory for query vector - FIXED TO 512
    __shared__ float s_query[512]; // CRITICAL CHANGE FOR 512

    // Cooperatively load query into shared memory.
    for (int i = tid_x + tid_y * blockDim.x; i < embed_dim; i += threads_per_block) {
        s_query[i] = __ldg(&query[i]);
    }
    __syncthreads(); 

    const int doc_id_base = blockIdx.x * block_height + tid_y;

    if (doc_id_base < num_docs) {
        float sum = 0.0f;
        const float* doc_ptr = &embeddings[doc_id_base * embed_dim];

        #pragma unroll
        for (int i = tid_x; i < embed_dim; i += block_width) {
            sum += __ldg(&doc_ptr[i]) * s_query[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid_x == 0) {
            similarities[doc_id_base] = sum;
        }
    }
}

/**
 * @brief Optimized CUDA kernel for computing dot products for a fixed dimension of 768.
 *
 * @tparam BLOCK_DIM_X Number of threads in X dimension.
 * @tparam BLOCK_DIM_Y Number of threads in Y dimension.
 */
template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 2)
gemv_optimized_768(
    const float* __restrict__ embeddings,
    const float* __restrict__ query,
    float* __restrict__ similarities,
    int num_docs,
    int embed_dim) {

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_width = BLOCK_DIM_X;
    const int block_height = BLOCK_DIM_Y;
    const int threads_per_block = blockDim.x * blockDim.y;

    __shared__ float s_query[768]; // FIXED TO 768

    for (int i = tid_x + tid_y * blockDim.x; i < embed_dim; i += threads_per_block) {
        s_query[i] = __ldg(&query[i]);
    }
    __syncthreads();

    const int doc_id_base = blockIdx.x * block_height + tid_y;

    if (doc_id_base < num_docs) {
        float sum = 0.0f;
        const float* doc_ptr = &embeddings[doc_id_base * embed_dim];

        #pragma unroll
        for (int i = tid_x; i < embed_dim; i += block_width) {
            sum += __ldg(&doc_ptr[i]) * s_query[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid_x == 0) {
            similarities[doc_id_base] = sum;
        }
    }
}

/**
 * @brief Optimized CUDA kernel for computing dot products for a fixed dimension of 1024.
 *
 * @tparam BLOCK_DIM_X Number of threads in X dimension.
 * @tparam BLOCK_DIM_Y Number of threads in Y dimension.
 */
template<int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 2)
gemv_optimized_1024(
    const float* __restrict__ embeddings,
    const float* __restrict__ query,
    float* __restrict__ similarities,
    int num_docs,
    int embed_dim) {

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_width = BLOCK_DIM_X;
    const int block_height = BLOCK_DIM_Y;
    const int threads_per_block = blockDim.x * blockDim.y;

    __shared__ float s_query[1024]; // FIXED TO 1024

    for (int i = tid_x + tid_y * blockDim.x; i < embed_dim; i += threads_per_block) {
        s_query[i] = __ldg(&query[i]);
    }
    __syncthreads();

    const int doc_id_base = blockIdx.x * block_height + tid_y;

    if (doc_id_base < num_docs) {
        float sum = 0.0f;
        const float* doc_ptr = &embeddings[doc_id_base * embed_dim];

        #pragma unroll
        for (int i = tid_x; i < embed_dim; i += block_width) {
            sum += __ldg(&doc_ptr[i]) * s_query[i];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid_x == 0) {
            similarities[doc_id_base] = sum;
        }
    }
}



/**
 * @brief Single-pass top-k kernel using thread-local and warp-level reduction.
 * Designed for small `top_k` (e.g., 5 or 8).
 * This kernel now assumes it can handle large `num_docs` by processing in a strided fashion.

 * @tparam BLOCK_SIZE The number of threads in a block.
 * @tparam K The number of top results to find (compile-time constant).
 * @param similarities Device pointer to all similarity scores.
 * @param out_indices Device pointer to output array for top K indices.
 * @param out_values Device pointer to output array for top K similarity values.
 * @param num_docs Total number of documents.
 */
template<int BLOCK_SIZE, int K>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
topk_kernel( // Renamed
    const float* __restrict__ similarities,
    int* __restrict__ out_indices,
    float* __restrict__ out_values,
    int num_docs) {
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 31; 
    const int warp_id = tid >> 5; 

    // Each thread maintains its own top-K list in registers.
    float thread_vals[K];
    int thread_indices[K];
    
    // Initialize thread-local top-K with smallest possible values.
    #pragma unroll
    for (int i = 0; i < K; i++) {
        thread_vals[i] = -FLT_MAX;
        thread_indices[i] = -1;
    }
    
    // Process strided elements across the similarities array.
    // Each thread processes elements with an offset of BLOCK_SIZE.
    for (int idx = tid; idx < num_docs; idx += BLOCK_SIZE) {
        float val = similarities[idx];
        
        // If current value is greater than the smallest in thread's top-K, insert it.
        if (val > thread_vals[K-1]) {
            thread_vals[K-1] = val;
            thread_indices[K-1] = idx;
            
            // Bubble up (insertion sort) to maintain sorted order.
            #pragma unroll
            for (int i = K-2; i >= 0; i--) {
                if (thread_vals[i] < thread_vals[i+1]) {
                    float tmp_val = thread_vals[i];
                    thread_vals[i] = thread_vals[i+1];
                    thread_vals[i+1] = tmp_val;
                    int tmp_idx = thread_indices[i];
                    thread_indices[i] = thread_indices[i+1];
                    thread_indices[i+1] = tmp_idx;
                } else {
                    break;
                }
            }
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { 
        #pragma unroll
        for (int i = 0; i < K; i++) {
            float other_val = __shfl_down_sync(0xffffffff, thread_vals[i], offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_indices[i], offset);
            
      
            if (lane_id < offset && other_val > thread_vals[K-1]) {
                thread_vals[K-1] = other_val;
                thread_indices[K-1] = other_idx;
                
                #pragma unroll
                for (int j = K-2; j >= 0; j--) {
                    if (thread_vals[j] < thread_vals[j+1]) {
                        float tmp_val_swap = thread_vals[j];
                        thread_vals[j] = thread_vals[j+1];
                        thread_vals[j+1] = tmp_val_swap;

                        int tmp_idx_swap = thread_indices[j]; 
                        thread_indices[j] = thread_indices[j+1];
                        thread_indices[j+1] = tmp_idx_swap;
                    }
                }
            }
        }
    }
    

    __shared__ float warp_vals[BLOCK_SIZE/32][K];
    __shared__ int warp_indices[BLOCK_SIZE/32][K];
    
    if (lane_id == 0) { 
        #pragma unroll
        for (int i = 0; i < K; i++) {
            warp_vals[warp_id][i] = thread_vals[i];
            warp_indices[warp_id][i] = thread_indices[i];
        }
    }
    
    __syncthreads(); 
    
    if (warp_id == 0 && lane_id < (BLOCK_SIZE/32)) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            thread_vals[i] = warp_vals[lane_id][i];
            thread_indices[i] = warp_indices[lane_id][i];
        }
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                float other_val = __shfl_down_sync(0xffffffff, thread_vals[i], offset);
                int other_idx = __shfl_down_sync(0xffffffff, thread_indices[i], offset);
                
                if (lane_id < offset && other_val > thread_vals[K-1]) {
                    thread_vals[K-1] = other_val;
                    thread_indices[K-1] = other_idx;
                    
                    #pragma unroll
                    for (int j = K-2; j >= 0; j--) {
                        if (thread_vals[j] < thread_vals[j+1]) {
                            float tmp_val_swap = thread_vals[j]; 
                            thread_vals[j] = thread_vals[j+1];
                            thread_vals[j+1] = tmp_val_swap;

                            int tmp_idx_swap = thread_indices[j]; 
                            thread_indices[j] = thread_indices[j+1];
                            thread_indices[j+1] = tmp_idx_swap;
                        }
                    }
                }
            }
        }
        
        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                out_indices[i] = thread_indices[i];
                out_values[i] = thread_vals[i];
            }
        }
    }
}

/**
 * @brief Initializes GPU resources for the search.
 * Allocates device memory for embeddings, query, similarities, and sorting.
 * Creates cuBLAS handle and CUDA stream.
 *
 * @param embeddings Host pointer to the initial document embeddings.
 * @param num_docs Number of documents.
 * @param embed_dim Dimension of embeddings.
 * @return 0 on success, non-zero on failure.
 */
extern "C" int init_gpu_search(const float* embeddings, int num_docs, int embed_dim) {
    if (g_initialized) {
        printf("Warning: GPU search already initialized. Cleaning up and re-initializing.\n");
        cleanup_gpu_search();
    }
    
    g_num_docs = num_docs;
    g_embed_dim = embed_dim;
    
    printf("Initializing GPU search: %d docs, %d dims\n", num_docs, embed_dim);
    
    cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    
    size_t embeddings_size = (size_t)num_docs * embed_dim * sizeof(float);
    size_t similarities_size = (size_t)num_docs * sizeof(float);
    size_t indices_size = (size_t)num_docs * sizeof(int);

    cudaCheck(cudaMalloc(&g_d_embeddings, embeddings_size));
    cudaCheck(cudaMalloc(&g_d_query, embed_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&g_d_similarities, similarities_size));
    cudaCheck(cudaMalloc(&g_d_original_indices, indices_size));
    

    cudaCheck(cudaMalloc(&g_d_cub_sorted_values, similarities_size));
    cudaCheck(cudaMalloc(&g_d_cub_sorted_indices, indices_size));

   
    cudaCheck(cudaMalloc(&g_d_temp_top_values, MAX_SUPPORTED_TOP_K_CUSTOM * sizeof(float)));
    cudaCheck(cudaMalloc(&g_d_temp_top_indices, MAX_SUPPORTED_TOP_K_CUSTOM * sizeof(int)));

    cudaCheck(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, g_cub_temp_storage_bytes,
        g_d_similarities, g_d_cub_sorted_values,
        g_d_original_indices, g_d_cub_sorted_indices,
        g_num_docs, 0, sizeof(float) * 8, g_stream));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMalloc(&g_d_cub_temp_storage, g_cub_temp_storage_bytes));

    cudaCheck(cudaMemcpy(g_d_embeddings, embeddings, embeddings_size, cudaMemcpyHostToDevice));
    
    cudaCheck(cudaStreamCreate(&g_stream));

    thrust::device_ptr<int> d_indices_ptr(g_d_original_indices);
    thrust::sequence(thrust::cuda::par.on(g_stream), d_indices_ptr, d_indices_ptr + num_docs);    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(g_stream)); 

    cublasCheck(cublasCreate(&g_cublas_handle)); 
    cublasCheck(cublasSetStream(g_cublas_handle, g_stream)); 
    
    g_initialized = true;
    return 0; 
}

/**
 * @brief Performs the similarity search and finds the top-k results.
 * This version uses a hybrid approach, prioritizing custom kernels for specific dimensions,
 * and falling back to cuBLAS for all other dimensions.
 *
 * @param query Host pointer to the query embedding.
 * @param indices Host pointer to store the resulting top-k indices.
 * @param similarities Host pointer to store the resulting top-k similarity scores.
 * @param top_k The number of top results to retrieve.
 */
extern "C" void gpu_search_topk(const float* query, int* indices, float* similarities, int top_k) {
    if (!g_initialized) {
        fprintf(stderr, "Error: GPU search not initialized. Call init_gpu_search() first.\n");
        return;
    }
    if (top_k <= 0) {
        fprintf(stderr, "Error: top_k must be a positive integer.\n");
        return;
    }
    if (top_k > g_num_docs) {
        fprintf(stderr, "Warning: top_k (%d) is greater than total documents (%d). Setting top_k = num_docs.\n", top_k, g_num_docs);
        top_k = g_num_docs;
    }
    
    cudaCheck(cudaMemcpyAsync(g_d_query, query, g_embed_dim * sizeof(float),    
                               cudaMemcpyHostToDevice, g_stream));
    
    // Step 1: Compute similarity scores (dot products)
    const int BLOCK_DIM_X = 32; 
    const int BLOCK_DIM_Y = 8;  
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    int grid_x = (g_num_docs + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
    dim3 grid_dims(grid_x);

    if (g_embed_dim == 384) {
        gemv_optimized_384<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid_dims, block_dims, 0, g_stream>>>(
            g_d_embeddings, g_d_query, g_d_similarities, g_num_docs, g_embed_dim);
    }    
    else if (g_embed_dim == 512) {
        gemv_optimized_512<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid_dims, block_dims, 0, g_stream>>>(
            g_d_embeddings, g_d_query, g_d_similarities, g_num_docs, g_embed_dim);
    }
    else if (g_embed_dim == 768) {
        gemv_optimized_768<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid_dims, block_dims, 0, g_stream>>>(
            g_d_embeddings, g_d_query, g_d_similarities, g_num_docs, g_embed_dim);
    }
    else if (g_embed_dim == 1024) {
        gemv_optimized_1024<BLOCK_DIM_X, BLOCK_DIM_Y><<<grid_dims, block_dims, 0, g_stream>>>(
            g_d_embeddings, g_d_query, g_d_similarities, g_num_docs, g_embed_dim);
    }
    else { 
        const float alpha = 1.0f, beta = 0.0f;
        // Perform GEMV using cuBLAS: C = alpha * op(A) * B + beta * C
        cublasCheck(cublasSgemv(g_cublas_handle, CUBLAS_OP_T, 
                                g_embed_dim, 
                                g_num_docs, 
                                &alpha,     
                                g_d_embeddings, 
                                g_embed_dim, 
                                g_d_query,  
                                1,          
                                &beta,      
                                g_d_similarities, 
                                1));         
    }
    cudaCheck(cudaGetLastError()); 
    
    // Step 2: Find top-k results
    if (top_k <= MAX_SUPPORTED_TOP_K_CUSTOM) {    

        int block_size = 1024; 
        
        topk_kernel<1024, MAX_SUPPORTED_TOP_K_CUSTOM><<<1, block_size, 0, g_stream>>>( 
            g_d_similarities, g_d_temp_top_indices, g_d_temp_top_values, g_num_docs);
        cudaCheck(cudaGetLastError()); 
        cudaCheck(cudaMemcpyAsync(indices, g_d_temp_top_indices, top_k * sizeof(int),    
                                   cudaMemcpyDeviceToHost, g_stream));
        if (similarities) {    
            cudaCheck(cudaMemcpyAsync(similarities, g_d_temp_top_values, top_k * sizeof(float),    
                                       cudaMemcpyDeviceToHost, g_stream));
        }
    }    
    // Fallback to CUB for larger top_k (i.e., top_k > 8 in this setup).
    else {    
        cudaCheck(cub::DeviceRadixSort::SortPairsDescending(    
            g_d_cub_temp_storage, g_cub_temp_storage_bytes,
            g_d_similarities, g_d_cub_sorted_values,        
            g_d_original_indices, g_d_cub_sorted_indices,    
            g_num_docs, 0, sizeof(float) * 8, g_stream));
        
        cudaCheck(cudaMemcpyAsync(indices, g_d_cub_sorted_indices, top_k * sizeof(int),    
                                   cudaMemcpyDeviceToHost, g_stream)); 
        if (similarities) {    
            cudaCheck(cudaMemcpyAsync(similarities, g_d_cub_sorted_values, top_k * sizeof(float),    
                                       cudaMemcpyDeviceToHost, g_stream));
        }
    }
    
    cudaCheck(cudaStreamSynchronize(g_stream));
}

extern "C" void cleanup_gpu_search() {
    if (g_initialized) {
        // Free device memory allocations
        if (g_d_embeddings) cudaCheck(cudaFree(g_d_embeddings));
        if (g_d_query) cudaCheck(cudaFree(g_d_query));
        if (g_d_similarities) cudaCheck(cudaFree(g_d_similarities));
        if (g_d_original_indices) cudaCheck(cudaFree(g_d_original_indices));
        if (g_d_cub_sorted_values) cudaCheck(cudaFree(g_d_cub_sorted_values));
        if (g_d_cub_sorted_indices) cudaCheck(cudaFree(g_d_cub_sorted_indices));
        if (g_d_cub_temp_storage) cudaCheck(cudaFree(g_d_cub_temp_storage));
        if (g_d_temp_top_values) cudaCheck(cudaFree(g_d_temp_top_values));    
        if (g_d_temp_top_indices) cudaCheck(cudaFree(g_d_temp_top_indices));    
        
        // Destroy cuBLAS handle and CUDA stream
        if (g_cublas_handle) cublasCheck(cublasDestroy(g_cublas_handle));    
        if (g_stream) cudaCheck(cudaStreamDestroy(g_stream));
        
        // Reset global state flags and sizes
        g_initialized = false;
        g_num_docs = 0;
        g_embed_dim = 0;
        g_cub_temp_storage_bytes = 0;

        // Reset pointers to nullptr
        g_d_embeddings = nullptr;
        g_d_query = nullptr;
        g_d_similarities = nullptr;
        g_d_original_indices = nullptr;
        g_d_cub_sorted_values = nullptr;
        g_d_cub_sorted_indices = nullptr;
        g_d_cub_temp_storage = nullptr;
        g_d_temp_top_values = nullptr;    
        g_d_temp_top_indices = nullptr;    
        g_cublas_handle = nullptr;
        g_stream = nullptr;

        printf("GPU resources cleaned up.\n");
    }
}
