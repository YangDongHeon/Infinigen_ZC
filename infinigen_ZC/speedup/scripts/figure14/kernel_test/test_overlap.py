import torch
import time
import gather_copy  # your built extension module exposing the gather_copy kernels
import torch.cuda.nvtx as nvtx

def run_tests():
    #--------------------------------------------------------------------------
    # Part 1: Setup for gather_copy kernels.
    #--------------------------------------------------------------------------
    # Dimensions for gather_copy.
    n = 2047
    nprime = 150
    b = 10
    h = 32
    d = 128

    # Allocate pinned CPU tensor for source (n, b, h, d).
    src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32) \
               .to(torch.bfloat16).pin_memory()
    # Allocate GPU tensor for destination (nprime, b, h, d).
    dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
    # Allocate GPU tensor for selection indices (nprime * b * h,).
    sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')

    # Get raw pointer addresses.
    src_ptr = int(src.data_ptr())
    dst_ptr = int(dst.data_ptr())
    sel_ptr = int(sel.data_ptr())

    #--------------------------------------------------------------------------
    # Part 2: Run a built-in CUDA kernel (matrix multiplication) on the default stream.
    #--------------------------------------------------------------------------
    # Create two large random matrices.
    A = torch.randn(2048, 2048, device='cuda')
    B = torch.randn(2048, 2048, device='cuda')
    
    gather_stream = torch.cuda.Stream()

    def time_kernel(kernel_func):
        with torch.cuda.stream(gather_stream):
            kernel_func(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        torch.matmul(A, B)
        time_kernel(gather_copy.gather_bf16_basic)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Time taken of basic: ", t1 - t0)

    
    torch.cuda.synchronize()
    nvtx.range_push("gather_bf16_basic")
    t0 = time.time()
    for _ in range(10):
        torch.matmul(A, B)
        time_kernel(gather_copy.gather_bf16_basic)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Time taken of basic: ", t1 - t0)
    nvtx.range_pop()

    torch.cuda.synchronize()
    nvtx.range_push("gather_bf16_2")
    t0 = time.time()
    for _ in range(10):
        torch.matmul(A, B)
        time_kernel(gather_copy.gather_bf16_2)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Time taken of 2: ", t1 - t0)
    nvtx.range_pop()


    torch.cuda.synchronize()
    nvtx.range_push("gather_bf16_basic")
    t0 = time.time()
    for _ in range(10):
        torch.matmul(A, B)
        time_kernel(gather_copy.gather_bf16_optimized)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Time taken of optimized: ", t1 - t0)
    nvtx.range_pop()

    


if __name__ == "__main__":
    run_tests()
