import torch

pred = torch.randn((4, 5))
print(pred)
values, indices = pred.topk(2, dim=1, largest=True, sorted=True)
print(indices)
# 用max得到的结果，设置keepdim为True，避免降维。因为topk函数返回的index不降维，shape和输入一致。
_, indices_max = pred.max(dim=1, keepdim=True)

print(indices_max == indices)
