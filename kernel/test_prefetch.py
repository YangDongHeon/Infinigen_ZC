import torch
import torch.nn.functional as F
import gather_copy  # Your compiled extension module from gather_copy.cu
import time
import torch.cuda.nvtx as nvtx

def select_tensor(prefetch_idx, tensor):
    """
    Gathers rows from the CPU tensor using torch.embedding.
    - prefetch_idx: tensor of shape (nprime, 1, bh)
    - tensor: a pinned CPU tensor of shape (n, bh, d)
    Returns a tensor of shape (nprime, bh, d) with dtype bfloat16.
    """
    nvtx.range_push("select_tensor")
    # Remove the extra dimension: shape becomes (nprime, bh)
    prefetch_idx = prefetch_idx.squeeze().to('cpu')
    bh = tensor.shape[1]
    # Compute flat indices: flat_index = token_index * bh + head_index.
    ind = prefetch_idx * bh + torch.arange(bh, device=tensor.device)[None, :]
    # Reshape tensor from (n, bh, d) to (n * bh, d)
    tensor_flat = tensor.reshape(-1, tensor.shape[2])
    # Gather the rows.
    selected = F.embedding(ind, tensor_flat)
    nvtx.range_pop()
    return selected

def prefetch_old(cpu_tensor, prefetch_idx, stream, dest, prefetch_buffer):
    """
    Old prefetch method (PyTorch-based) modified to include an extra CPU-to-CPU copy.
    This version does not allocate the CPU prefetch buffer internally; instead, it uses
    the externally provided prefetch_buffer.
    
      - cpu_tensor: pinned CPU tensor of shape (n, bh, d) with dtype bfloat16.
      - prefetch_idx: tensor of shape (nprime, 1, bh).
      - dest: preallocated GPU tensor of shape (nprime, bh, d) into which data is copied.
      - prefetch_buffer: preallocated pinned CPU tensor of shape (2, max_num_kv, bh, d).
    
    The function gathers the data using select_tensor, copies it into the prefetch_buffer,
    and then copies from that buffer into dest.
    """
    nvtx.range_push("prefetch_old_select")
    # Gather the selected rows (shape: (nprime, bh, d))
    selected = select_tensor(prefetch_idx, cpu_tensor)
    nvtx.range_pop()
    
    nprime = prefetch_idx.shape[0]
    # Simulate extra CPU-to-CPU copy by writing the gathered result into prefetch_buffer.
    nvtx.range_push("cpu_to_cpu_copy")
    prefetch_buffer[0, :nprime] = selected
    nvtx.range_pop()
    
    nvtx.range_push("prefetch_old_copy")
    # Copy from the prefetch buffer (first slice) into the provided GPU tensor 'dest'.
    with torch.cuda.stream(stream):
        dest.copy_(prefetch_buffer[0, :nprime], non_blocking=True)
    nvtx.range_pop()
    
    torch.cuda.synchronize()
    return dest

def test_compare():
    # Dimensions.
    n = 2047       # number of tokens in the source tensor
    nprime = 59   # number of prefetch rows (output tokens)
    b = 4         # batch dimension (or similar grouping)
    h = 32         # number of heads; so bh = b * h = 320
    d = 128        # feature dimension
    bh = b * h

    # ---------------------------
    # Create Source Tensor (bfloat16)
    # ---------------------------
    # Create a source tensor in shape (n, b, h, d) on CPU with dtype bfloat16.
    src = torch.randint(0, 100, (n, b, h, d), dtype=torch.int32).to(torch.bfloat16).pin_memory()
    print("src : ", src)
    # For the old method, view src as shape (n, bh, d)
    src_old = src.view(n, bh, d)

    # ---------------------------
    # Create Selection Arrays (common to both methods)
    # ---------------------------
    # Create a selection array on GPU with shape (nprime, bh) (each element in [0, n)).
    sel_common = torch.randint(0, n, (nprime, bh), dtype=torch.int64, device='cuda')
    print("sel_common[0] : ", sel_common[0])
    # For the old method, unsqueeze to shape (nprime, 1, bh)
    prefetch_idx = sel_common.unsqueeze(1)
    # For the new gather kernel, flatten the selection array to shape (nprime * bh,)
    sel_new = sel_common.flatten()

    # ---------------------------
    # Allocate Destination Tensors
    # ---------------------------
    # Allocate a destination tensor for the new gather kernel: shape (nprime, b, h, d)
    dst_new = torch.empty((nprime, b, h, d), dtype=torch.bfloat16, device='cuda')
    # Allocate an empty tensor for the old method (shape: (nprime, bh, d))
    dst_old = torch.empty((nprime, bh, d), dtype=torch.bfloat16, device='cuda')

    # ---------------------------
    # Allocate External Prefetch Buffer (for extra CPU-to-CPU copy)
    # ---------------------------
    # This buffer simulates self.prefetch_kv.data with shape (2, max_num_kv, bh, d)
    max_num_kv = 400  # example capacity
    prefetch_buffer = torch.empty((2, max_num_kv, bh, d), dtype=torch.bfloat16, device='cpu').pin_memory()

    # ---------------------------
    # Create a CUDA stream for asynchronous operations.
    # ---------------------------
    stream = torch.cuda.Stream()

    # ---------------------------
    # Warm-up Both Methods.
    # ---------------------------
    for _ in range(5):
        _ = prefetch_old(src_old, prefetch_idx, stream, dst_old, prefetch_buffer)
        src_ptr = int(src.data_ptr())      # src: shape (n, b, h, d)
        dst_ptr = int(dst_new.data_ptr())    # dst_new: shape (nprime, b, h, d)
        sel_ptr = int(sel_new.data_ptr())    # sel_new: shape (nprime * bh,)
        gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,128,256)

    # ---------------------------
    # Final Verification of Outputs.
    # ---------------------------
    src_ptr = int(src.data_ptr())      # src: shape (n, b, h, d)
    dst_ptr = int(dst_new.data_ptr())    # dst_new: shape (nprime, b, h, d)
    sel_ptr = int(sel_new.data_ptr())    # sel_new: shape (nprime * bh,)
    out_old = prefetch_old(src_old, prefetch_idx, stream, dst_old, prefetch_buffer)  # shape: (nprime, bh, d)
    gather_copy.gather_bf16_optimized(src_ptr, dst_ptr, sel_ptr, n, nprime, b, h, d,128,256)
    out_new = dst_new.view(nprime, bh, d)  # reshape new output to (nprime, bh, d)
    print("out_old[0] : ", out_old[0])
    print("out_new[0] : ", out_new[0]) 
    diff = (out_old.to(torch.float32) - out_new.to(torch.float32)).abs().sum().item()
    if diff == 0:
        print("Final outputs match!")
    else:
        print(f"Final outputs do not match, total difference = {diff}")

if __name__ == '__main__':
    test_compare()
