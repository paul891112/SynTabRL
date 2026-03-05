import torch
import torch.nn as nn

# 1. Create a dummy input tensor
# Let's say we have 3 "scores" (e.g., distances)
input_tensor = torch.tensor([[1.0, 1.0, 1e-30],
                             [1.0, 1e-16, 1.0],
                             [1e-30, 1.0, 1.0]])

# 2. Initialize the Softmin layer
# dim=0 means we apply it across the first dimension
output = nn.Softmin(dim=1)(input_tensor)

lse = -0.01 * torch.logsumexp(-input_tensor / 0.01, dim=1)

print(f"Input:  {input_tensor}")
print(f"Output: {output}")
print(f"Logsumexp: {lse}")
print(f"Sum:    {output.sum()}") # Always sums to 1.0