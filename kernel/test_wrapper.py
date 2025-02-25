import torch
import gather_copy  # Import the built extension module

# Example dimensions.
n = 2047
nprime = 150
b = 10
h = 32
d = 128

# Allocate buffers:
# src: pinned CPU tensor of shape (n, b, h, d)
src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
src_ptr = int(src.data_ptr())
dst_ptr = int(dst.data_ptr())
sel_ptr = int(sel.data_ptr())

# Launch the kernel via the extension module.
for i in range(5):
    src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
    dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
    sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
    src_ptr = int(src.data_ptr())
    dst_ptr = int(dst.data_ptr())
    sel_ptr = int(sel.data_ptr())

    gather_copy.gather_bf16_basic(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,0,16)
    src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
    dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
    sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
    src_ptr = int(src.data_ptr())
    dst_ptr = int(dst.data_ptr())
    sel_ptr = int(sel.data_ptr())

    gather_copy.gather_bf16_2(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,0,256)
    src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
    dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
    sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
    src_ptr = int(src.data_ptr())
    dst_ptr = int(dst.data_ptr())
    sel_ptr = int(sel.data_ptr())

    gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,128,256)

for i in range(6):
    for j in range(3):
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())
        
        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,4,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,8,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,16,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,32,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,64,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,128,16*(2**i))
        src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
        dst = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
        sel = torch.randint(0, n, (nprime * b * h,), dtype=torch.int64, device='cuda')
        src_ptr = int(src.data_ptr())
        dst_ptr = int(dst.data_ptr())
        sel_ptr = int(sel.data_ptr())

        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,256,16*(2**i))

print("Data gathering completed; check dst for the results.")
