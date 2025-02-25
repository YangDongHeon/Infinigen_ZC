import numpy as np
import torch

try:
    print("NumPy version:", np.__version__)
    print("Torch version:", torch.__version__)
    tensor = torch.tensor([1, 2, 3])
    array = tensor.cpu().detach().numpy()
    print("Conversion to NumPy array successful:", array)
except Exception as e:
    print("Error:", e)

