# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchviz import make_dot


# N = 100
# X = torch.randn(N)  # Input data
# Y = 2 * X + 1 + torch.randn(N) * 0.5  # Generating output data with noise


# ## Defining the computational graph
# # data and parameters, parameter initialization
# a = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)

# # X = torch.tensor([1, 2, 3, 4, 5])
# # Y = torch.tensor([3, 5, 7, 9, 11])

# # N = X.shape[0]
# # print(N)
# # optimization procedure: gradient descent
# optimizer = optim.SGD([a, b], lr=0.01)

# for i in range(2000):
#     # affine regression model
#     Y_ = a*X + b

#     diff = (Y-Y_)

#     # quadratic loss
#     loss = torch.sum(diff**2) / N

#     # gradient calculation
#     loss.backward()

#     # make_dot(Y_, params=dict(a=a, b=b)).render("attached", format="png")

#     # optimization step
#     optimizer.step()
#     print(a.grad)
#     print(b.grad)
#     # gradient reset
#     optimizer.zero_grad()

#     print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')


# import torch

# def f(x, a, b):
#     return a*x + b


# a = torch.tensor(5., requires_grad=True)
# b = torch.tensor(8., requires_grad=True)
# x = torch.tensor(2.)

# y = f(x, a, b)
# s = a ** 2

# (y + s).backward()

# print(f"y: {y}, g_a: {a.grad}, g_b: {b.grad}")

import numpy as np
import torch

# a = torch.randn(3, 4)
# b = torch.tensor([[True, False, False, False],
#                   [False, True, False, False],
#                   [False, False, True, False]])

# print(a[b])

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

idx = torch.randperm(len(a))
print(idx)
for i in range(len(a)//3):
    print(a[idx[i*3:(i+1)*3]])