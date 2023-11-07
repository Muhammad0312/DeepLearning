import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot


N = 100
X = torch.randn(N)  # Input data
Y = 2 * X + 1 + torch.randn(N) * 0.5  # Generating output data with noise


## Defining the computational graph
# data and parameters, parameter initialization
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# X = torch.tensor([1, 2, 3, 4, 5])
# Y = torch.tensor([3, 5, 7, 9, 11])

# N = X.shape[0]
# print(N)
# optimization procedure: gradient descent
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(2000):
    # affine regression model
    Y_ = a*X + b

    diff = (Y-Y_)

    # quadratic loss
    loss = torch.sum(diff**2) / N

    # gradient calculation
    loss.backward()

    # make_dot(Y_, params=dict(a=a, b=b)).render("attached", format="png")

    # optimization step
    optimizer.step()
    print(a.grad)
    print(b.grad)
    # gradient reset
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')


