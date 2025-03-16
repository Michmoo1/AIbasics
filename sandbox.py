import torch
import torch.nn.functional as F

# Checking activation functions
x = torch.tensor([-3, 0, 3, 4])

x_f = x.to(torch.float)

# Leaky ReLu
print(f"Leaky ReLu: {F.leaky_relu(x_f, negative_slope=0.01)}")

#Tanh
print(f"Tanh: {F.tanh(x_f)}")


