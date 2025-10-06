import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))

# Test du gradient de Linear
x = torch.randn(10, 13, dtype=torch.float64, requires_grad=True)
w = torch.randn(13, 3, dtype=torch.float64, requires_grad=True)
b = torch.randn(3, dtype=torch.float64, requires_grad=True)

torch.autograd.gradcheck(linear, (x, w, b))

# Test du gradient de Linear
x = torch.randn(10, 13, dtype=torch.float64, requires_grad=True)  # batch de 10, 13 features
w = torch.randn(13, 3, dtype=torch.float64, requires_grad=True)   # poids 13 -> 3
b = torch.randn(3, dtype=torch.float64, requires_grad=True)       # bias pour 3 sorties

torch.autograd.gradcheck(linear, (x, w, b))
