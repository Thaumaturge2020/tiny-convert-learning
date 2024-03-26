import torch

# 创建一个示例张量
x = torch.tensor([[1, 2], [3, 4]])

# 在指定维度上重复张量
repeated_x = x.tile(2, 3)  # 在维度0上重复2次，在维度1上重复3次

print("原始张量:")
print(x)

print("\n重复张量:")
print(repeated_x)