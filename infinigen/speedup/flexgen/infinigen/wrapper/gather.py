import ctypes
import numpy as np
import time

# Load the shared library.
lib = ctypes.CDLL('./libgather.so')

# Define argument and return types for the functions.
lib.init_gather_workers.argtypes = [ctypes.c_int]
lib.shutdown_gather_workers.argtypes = []
lib.start_gather_task.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # src
    ctypes.POINTER(ctypes.c_float),  # dst
    ctypes.POINTER(ctypes.c_int),    # sel
    ctypes.c_int                     # num_sel
]
lib.start_gather_task.restype = None
lib.wait_for_gather_task.argtypes = []
# Initialize 4 worker threads.
lib.init_gather_workers(8)

# Create sample data.
num = 1000000000
num_sel = 1000000
src_np = np.arange(num, dtype=np.float32)
dst_np = np.empty(num_sel, dtype=np.float32)
sel_np = np.random.randint(0, num, size=num_sel, dtype=np.int32)

# Get pointers from the numpy arrays.
src_ptr = src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dst_ptr = dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
sel_ptr = sel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

# Signal a gather task.
start_time = time.perf_counter()
print("start_time: ", start_time)   
lib.start_gather_task(src_ptr, dst_ptr, sel_ptr, num_sel)
print("oh yea~~ im free")
lib.wait_for_gather_task()
end_time = time.perf_counter()
print("end_time: ", end_time)
elapsed_time = (end_time - start_time) * 1e3  # Convert to microseconds
print("Gather task took %.2f ms." % elapsed_time)

print("Gather task completed. First few results:")
print(dst_np[:10])

# Shutdown the worker threads.
lib.shutdown_gather_workers()
