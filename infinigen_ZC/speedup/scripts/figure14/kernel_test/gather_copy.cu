#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>  // ATen's CUDA utilities

namespace py = pybind11;

// Use bf16_t as a plain 16-bit type.
using bf16_t = short;

// Define VECTOR_WIDTH once (each int4 loads 16 bytes = 8 bf16_t elements)
#ifndef VECTOR_WIDTH
#define VECTOR_WIDTH 8
#endif

//=============================================================================
// Kernel 1: Basic kernel (one block per output row)
//=============================================================================
template <typename T>
__global__ void gather_bf16_kernel_block(
    const T * __restrict__ h_src,  // source tensor in pinned host memory
    T * __restrict__ d_dst,        // destination tensor in device memory
    const int64_t * __restrict__ sel, // selection array (length: nprime * b * h)
    int n, int nprime, int b, int h, int d)
{
    int out_row = blockIdx.x;
    int rem = out_row % (b * h);
    int ib = rem / h;
    int ih = rem % h;
    int src_n = sel[out_row];
    
    int src_offset = ((src_n * b + ib) * h + ih) * d;
    int dst_offset = out_row * d;
    
    int num_vec = d / VECTOR_WIDTH;
    const int4 *src_vec = reinterpret_cast<const int4*>(h_src + src_offset);
    int4 *dst_vec = reinterpret_cast<int4*>(d_dst + dst_offset);
    
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
}

void gather_bf16_wrapper_basic(uintptr_t src_ptr, uintptr_t dst_ptr, uintptr_t sel_ptr,
                               int n, int nprime, int b, int h, int d, int gridSize, int blockSize) {
    bf16_t* h_src = reinterpret_cast<bf16_t*>(src_ptr);
    bf16_t* d_dst = reinterpret_cast<bf16_t*>(dst_ptr);
    const int64_t* sel = reinterpret_cast<const int64_t*>(sel_ptr);
    
    bf16_t* d_src;
    cudaHostGetDevicePointer((void**)&d_src, h_src, 0);
    
    // Launch one block per output row.
    int total_rows = nprime * b * h;
    int gridSize_real = total_rows;
    int blockSize_real = 16; // e.g., 16 threads per block.
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_bf16_kernel_block<bf16_t><<<gridSize_real, blockSize_real, 0, stream>>>(d_src, d_dst, sel,
                                                                          n, nprime, b, h, d);
}

//=============================================================================
// Kernel 2: Grid-stride loop kernel using thread groups
//=============================================================================
template <typename T>
__global__ void gather_bf16_kernel_block_2(
    const T * __restrict__ h_src,  // source tensor in pinned host memory
    T * __restrict__ d_dst,        // destination tensor in device memory
    const int64_t * __restrict__ sel, // selection array (length: nprime * b * h)
    int n, int nprime, int b, int h, int d)
{
    // Each block handles 16 rows.
    const int rows_per_block = 16;
    // Divide 256 threads into 16 groups of 16 threads.
    int group_id = threadIdx.x / 16;  // Which row in this block (0..15)
    int lane_id  = threadIdx.x & 15;    // Which thread within that row
    
    // Compute global output row index.
    int out_row = blockIdx.x * rows_per_block + group_id;
    int total_rows = nprime * b * h;
    if (out_row >= total_rows) return;
    
    // Calculate b and h indices for this row.
    int rem = out_row % (b * h);
    int ib = rem / h;
    int ih = rem % h;
    int src_n = sel[out_row];
    
    int src_offset = ((src_n * b + ib) * h + ih) * d;
    int dst_offset = out_row * d;
    
    int num_vec = d / VECTOR_WIDTH;
    const int4* src_vec = reinterpret_cast<const int4*>(h_src + src_offset);
    int4* dst_vec = reinterpret_cast<int4*>(d_dst + dst_offset);
    
    // Each thread in the group copies a portion of the row.
    for (int i = lane_id; i < num_vec; i += 16) {
        dst_vec[i] = src_vec[i];
    }
}

void gather_bf16_wrapper_2(uintptr_t src_ptr, uintptr_t dst_ptr, uintptr_t sel_ptr,
                                  int n, int nprime, int b, int h, int d,int gridSize, int blockSize) {
    bf16_t* h_src = reinterpret_cast<bf16_t*>(src_ptr);
    bf16_t* d_dst = reinterpret_cast<bf16_t*>(dst_ptr);
    const int64_t* sel = reinterpret_cast<const int64_t*>(sel_ptr);
    
    bf16_t* d_src;
    cudaHostGetDevicePointer((void**)&d_src, h_src, 0);
    
    int total_rows = nprime * b * h;
    const int rows_per_block = 16;
    int gridSize_real = (total_rows + rows_per_block - 1) / rows_per_block;  // one block covers 16 rows
    int blockSize_real = 256;  // 16 rows * 16 threads per row
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_bf16_kernel_block_2<bf16_t>
        <<<gridSize_real, blockSize_real, 0, stream>>>(d_src, d_dst, sel, n, nprime, b, h, d);
}


//=============================================================================
// Kernel 3: Ultra-optimized kernel using bit-wise arithmetic and __ldg()
//=============================================================================
template <typename T>
__global__ void gather_bf16_kernel_block_optimized(
    const T * __restrict__ h_src,   // source tensor in pinned host memory
    T * __restrict__ d_dst,         // destination tensor in device memory
    const int64_t * __restrict__ sel,  // selection array (length: nprime * b * h)
    int n, int nprime, int b, int h, int d)
{
    // Here d is fixed to 128, so num_vec should be 128 / VECTOR_WIDTH.
    //const int num_vec = 128 / VECTOR_WIDTH;
    
    int total_rows = nprime * b * h;
    int bh = b * h;
    
    // Divide the 256 threads in the block into groups of 16.
    int group_id = threadIdx.x >> 4;    // equivalent to threadIdx.x / 16
    int lane_id  = threadIdx.x & 15;      // equivalent to threadIdx.x % 16
    
    int groups_per_block = blockDim.x >> 4; // blockDim.x / 16 (should be 16 for 256 threads)
    int global_group_id = blockIdx.x * groups_per_block + group_id;
    int total_groups = gridDim.x * groups_per_block;
    
    // Grid-stride loop over rows assigned to each group.
    for (int out_row = global_group_id; out_row < total_rows; out_row += total_groups) {
        int rem = out_row % bh;
        int ib = rem / h;
        int ih = rem % h;
        
        // Use __ldg to load sel[out_row] from read-only cache.
        int src_n = __ldg(&sel[out_row]);
        int src_offset = ((src_n * b + ib) * h + ih) * d;
        int dst_offset = out_row * d;

        const int4* src_vec = reinterpret_cast<const int4*>(h_src + src_offset);
        int4* dst_vec = reinterpret_cast<int4*>(d_dst + dst_offset);
        
        // Each lane in the group copies exactly one vector element.
        dst_vec[lane_id] = src_vec[lane_id];
    }
}

void gather_bf16_wrapper_optimized(uintptr_t src_ptr, uintptr_t dst_ptr, uintptr_t sel_ptr,
                                   int n, int nprime, int b, int h, int d, int gridSize, int blockSize) {
    bf16_t* h_src = reinterpret_cast<bf16_t*>(src_ptr);
    bf16_t* d_dst = reinterpret_cast<bf16_t*>(dst_ptr);
    const int64_t* sel = reinterpret_cast<const int64_t*>(sel_ptr);
    
    bf16_t* d_src;
    cudaHostGetDevicePointer((void**)&d_src, h_src, 0);
    
    // Launch with 16 blocks of 256 threads.
    //int gridSize = 32;
    //int blockSize = 256;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_bf16_kernel_block_optimized<bf16_t><<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, sel,
                                                                                    n, nprime, b, h, d);
}

//=============================================================================
// Pybind11 module exposing all three variants
//=============================================================================
PYBIND11_MODULE(gather_copy, m) {
    m.doc() = "Gather copy CUDA data mover extension using ATen to get the current CUDA stream";
    
    m.def("gather_bf16_basic", &gather_bf16_wrapper_basic,
          "Launch the basic gather_bf16 kernel on the current CUDA stream",
          py::arg("src_ptr"), py::arg("dst_ptr"), py::arg("sel_ptr"),
          py::arg("n"), py::arg("nprime"), py::arg("b"), py::arg("h"), py::arg("d"), py::arg("gridSize"), py::arg("blockSize"));
    
    m.def("gather_bf16_2", &gather_bf16_wrapper_2,
          "Launch the grid-stride gather_bf16 kernel (kernel 2) on the current CUDA stream",
          py::arg("src_ptr"), py::arg("dst_ptr"), py::arg("sel_ptr"),
          py::arg("n"), py::arg("nprime"), py::arg("b"), py::arg("h"), py::arg("d"), py::arg("gridSize"), py::arg("blockSize"));
    
    m.def("gather_bf16_optimized", &gather_bf16_wrapper_optimized,
          "Launch the ultra-optimized gather_bf16 kernel on the current CUDA stream",
          py::arg("src_ptr"), py::arg("dst_ptr"), py::arg("sel_ptr"),
          py::arg("n"), py::arg("nprime"), py::arg("b"), py::arg("h"), py::arg("d"), py::arg("gridSize"), py::arg("blockSize"));
}
