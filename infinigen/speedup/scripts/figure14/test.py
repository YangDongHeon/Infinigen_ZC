import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    
    # Get the GPU device name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")

    # Allocate a tensor on the GPU
    try:
        device = torch.device("cuda")  # Use the first GPU
        x = torch.rand(1000, 1000, device=device)  # Create a random tensor
        y = torch.matmul(x, x)  # Perform a simple operation
        print("Tensor operation on GPU was successful!")
    except Exception as e:
        print(f"Error during tensor operation on GPU: {e}")
else:
    print("GPU is not available. Please check your setup.")
