import torch

# 1. 判断 CUDA GPU 是否可用（NVIDIA 显卡）
has_cuda = torch.cuda.is_available()
print(f"CUDA GPU 可用: {has_cuda}")

# 2. 若为苹果 Silicon 芯片，判断 MPS 是否可用
has_mps = torch.backends.mps.is_available()
print(f"Apple MPS GPU 可用: {has_mps}")

# 3. 自动选择最优设备
device = torch.device(
    "cuda:0" if has_cuda 
    else "mps" if has_mps 
    else "cpu"
)
print(f"最终选择的设备: {device}")

# 4. 进阶：查看 CUDA 设备详情（若有）
if has_cuda:
    print(f"可用 CUDA 设备数: {torch.cuda.device_count()}")
    print(f"第 0 块 GPU 名称: {torch.cuda.get_device_name(0)}")