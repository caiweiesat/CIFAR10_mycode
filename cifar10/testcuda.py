import torch
# 1. PyTorch编译时绑定的CUDA版本（安装torch时匹配的）
print(torch.version.cuda)
# 2. 判断GPU是否可用
print(torch.cuda.is_available())