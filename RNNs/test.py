import torch
import torch.nn as nn
# Assuming your data tensor has dimensions (10, 35, 300)
# data_tensor = torch.randn(10, 35, 300)
# print(data_tensor)

# Apply mean pooling along the temporal dimension (axis 1)
# mean_pooled_tensor = nn.AdaptiveAvgPool1d(1)(data_tensor.permute(0,2,1)).squeeze()

# # The resulting tensor will have dimensions (10, 300)
# print(mean_pooled_tensor.shape)


embeddings = torch.empty(10, 10).normal_(mean=0, std=1)
print(embeddings)

mbeddings = torch.randn(10, 10)
print(embeddings)