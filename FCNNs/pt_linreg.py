import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot


N = 1
X = torch.randn(N)  # Input data
Y = 2 * X + 1 + torch.randn(N) * 0.5  # Generating output data with noise


X = torch.tensor([1, 2, 3])
Y = torch.tensor([3, 5, 7])
# print('X: ', X)
# print('Y: ', Y)


## Defining the computational graph
# data and parameters, parameter initialization
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# print('a: ', a)
# print('b: ', b)

# optimization procedure: gradient descent
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(1):
    # affine regression model
    Y_ = a*X + b

    diff = (Y-Y_)

    # quadratic loss
    loss = torch.sum(diff**2) / N

    grad_a = sum(2*(Y-Y_)*-X)
    grad_b = sum(2*(Y-Y_)*-1)
    print('Gradient wrt to a: ',grad_a)
    print('Gradient wrt to b: ',grad_b)

    # gradient calculation
    loss.backward()

    # make_dot(Y_, params=dict(a=a, b=b)).render("attached", format="png")

    # optimization step
    optimizer.step()
    print('Grad a: ',a.grad)
    print('Grad b: ',b.grad)
    # gradient reset
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')