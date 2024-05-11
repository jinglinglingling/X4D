import torch
from einops import rearrange
i_tensor = torch.randn(16, 3, 224, 224)		# 在CV中很常见的四维tensor： （N，C，H，W）

print(type(i_tensor))
print(i_tensor.shape)
o_tensor = rearrange(i_tensor, 'n c h w -> n h w c')
print(o_tensor.shape)

