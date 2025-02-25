#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// Use bf16_t as a plain 16-bit type.
using bf16_t = short;

// Each vectorized load (int4) loads 16 bytes = 8 bf16_t elements.
#define VECTOR_WIDTH 8  
// We now use BLOCK_SIZE = 16 because 128/8 = 16 vector loads per row.
#define BLOCK_SIZE 16

// Kernel that cooperatively copies one row per block.
// Each output row corresponds to a (b, h) slice in the output tensor.
// The selection array 'sel' (of shape: nprime * b * h) indicates which n-index from
// the source tensor (shape: n,b,h,d) should be copied.
template <typename T>
__global__ void gather_bf16_kernel_block(
    const T * __restrict__ h_src, // pinned host memory (mapped) pointer for source tensor
    T * __restrict__ d_dst,       // destination GPU tensor pointer
    const int * __restrict__ sel, // selection array (shape: nprime * b * h)
    int n, int nprime, int b, int h, int d)
{
    // Each block is responsible for one output row.
    int out_row = blockIdx.x;
    //int total_rows = nprime * b * h;
    //if (out_row >= total_rows) return;

    // Decode the output row index into (i_nprime, ib, ih)
    int rem = out_row % (b * h);
    int ib = rem / h;
    int ih = rem % h;

    // Get the selected source n-index for this row.
    int src_n = sel[out_row];

    // Compute offsets.
    // Source tensor is stored in (n,b,h,d) order.
    int src_offset = ((src_n * b + ib) * h + ih) * d;
    // Destination tensor is stored in (nprime,b,h,d) order.
    int dst_offset = out_row * d;

    // Use vectorized copying by reinterpreting pointers as int4.
    // Since each bf16_t is 2 bytes, an int4 (16 bytes) corresponds to 8 bf16_t elements.
    const int num_vec = d / VECTOR_WIDTH;  // For d==128, num_vec == 16.
    const int4 *src_vec = reinterpret_cast<const int4*>(h_src + src_offset);
    int4 *dst_vec = reinterpret_cast<int4*>(d_dst + dst_offset);

    // Each block’s threads cooperatively copy the vector loads.
    // With BLOCK_SIZE set to 16, each thread will handle exactly one vector load.
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
}

//
// Test code
//
int main()
{
    // ---------------- Dimensions ----------------
    // Host tensor shape: (n, b, h, d)
    // GPU output tensor shape: (nprime, b, h, d)
    int n = 8000;       // source n dimension
    int nprime = 150;  // output n' dimension
    int b = 10;
    int h = 32;
    int d = 128;     // fixed

    size_t num_src = static_cast<size_t>(n) * b * h * d;
    size_t num_dst = static_cast<size_t>(nprime) * b * h * d;
    size_t bytes_src = num_src * sizeof(bf16_t);
    size_t bytes_dst = num_dst * sizeof(bf16_t);
    size_t bytes_sel = (nprime * b * h) * sizeof(int);

    // -------- Allocate pinned host memory for the source tensor --------
    bf16_t *h_src;
    cudaError_t err = cudaHostAlloc((void**)&h_src, bytes_src, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // Get the device pointer for the pinned memory.
    bf16_t *d_h_src;
    err = cudaHostGetDevicePointer((void**)&d_h_src, h_src, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostGetDevicePointer failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // -------- Initialize the source tensor with a known pattern --------
    // Here we simply fill with i % 100.
    for (size_t i = 0; i < num_src; ++i) {
        h_src[i] = static_cast<bf16_t>(i % 100);
    }

    // -------- Allocate GPU memory for the destination tensor --------
    bf16_t *d_dst;
    err = cudaMalloc((void**)&d_dst, bytes_dst);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_dst failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // -------- Allocate GPU memory for the selection indices --------
    int *d_sel;
    err = cudaMalloc((void**)&d_sel, bytes_sel);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_sel failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // Create and fill the selection array on host.
    int *h_sel = new int[nprime * b * h];
    for (int i = 0; i < nprime * b * h; i++) {
         h_sel[i] = rand() % n;  // randomly select a valid n index for each output row
    }
    err = cudaMemcpy(d_sel, h_sel, bytes_sel, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         std::cerr << "cudaMemcpy for d_sel failed: " << cudaGetErrorString(err) << std::endl;
         return -1;
    }

    // -------- Launch the kernel --------
    // We launch one block per output row.
    int total_rows = nprime * b * h;
    dim3 grid(total_rows);
    // Using BLOCK_SIZE = 16 as requested.
    dim3 block(BLOCK_SIZE);

    gather_bf16_kernel_block<bf16_t><<<grid, block>>>(d_h_src, d_dst, d_sel,
                                                        n, nprime, b, h, d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
         std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
         return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
         std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
         return -1;
    }

    // -------- Copy back the destination tensor for verification --------
    bf16_t *h_dst = new bf16_t[num_dst];
    err = cudaMemcpy(h_dst, d_dst, bytes_dst, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
         std::cerr << "cudaMemcpy for d_dst failed: " << cudaGetErrorString(err) << std::endl;
         return -1;
    }

    // -------- Verify the results --------
    // Each output row should match the source row indicated by h_sel.
    bool pass = true;
    for (int out_row = 0; out_row < total_rows; out_row++) {
        // Decode (i_nprime, ib, ih) from out_row.
        int rem = out_row % (b * h);
        int ib = rem / h;
        int ih = rem % h;
        int src_n = h_sel[out_row];  // selected n index for this row

        int src_offset = ((src_n * b + ib) * h + ih) * d;
        int dst_offset = out_row * d;
        for (int i = 0; i < d; i++) {
            short src_val = h_src[src_offset + i];
            short dst_val = h_dst[dst_offset + i];
            if (src_val != dst_val) {
                std::cerr << "Mismatch at out_row " << out_row 
                          << ", element " << i << ": expected " 
                          << src_val << ", got " << dst_val << std::endl;
                pass = false;
                break;
            }
        }
        if (!pass) break;
    }
    std::cout << (pass ? "Test PASSED" : "Test FAILED") << std::endl;

    // -------- Cleanup --------
    cudaFreeHost(h_src);
    cudaFree(d_dst);
    cudaFree(d_sel);
    delete[] h_sel;
    delete[] h_dst;

    return 0;
}
