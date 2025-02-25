nvcc copy_kernel.cu -o copy_kernel

nsys profile --stats=true ./copy_kernel
