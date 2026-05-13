import time

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)

# 简单的 GPU 计算测试

# torch.cuda.set_per_process_memory_fraction(0.1, 0)
a = torch.ones(10000, 10000, device='cuda')
b = torch.ones(10000, 10000, device='cuda')
# a = torch.ones(7000, 7000, device='cuda')
# b = torch.ones(7000, 7000, device='cuda')
c = torch.ones(10000, 10000, device='cpu')
d = torch.ones(10000, 10000, device='cpu')
GPU_start = time.time()
e = a @ b
GPU_end = time.time()
CPU_start = time.time()
f = c @ d
CPU_end = time.time()

print("GPU computation result shape:", e.shape, GPU_end - GPU_start)
print("CPU computation result shape:", f.shape, CPU_end - CPU_start)