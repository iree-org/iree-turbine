import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'

# [ y = W_n * x_n + W_{n-1} * x_{n-1} + ... + W_1 * x_1 + b ]
torch.cuda.manual_seed_all(0)
x = torch.linspace(-1, 1, 100).reshape(-1)
y = 3 * x + 2 + torch.randn(x.size()) * 0.2

# cvt to tensor
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)
print(x)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        print(self.weight)
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x : torch.Tensor):
        out =  x * self.weight + self.bias
        return out


# model = SimpleMLP().to(device)
mod = SimpleMLP().to(device)

model = torch.compile(mod, backend='turbine_cpu')

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

epochs = 2000
for epoch in range(epochs):
    y_pred = model(x)
    # print(y_pred)
    
    loss = loss_func(y_pred.to(device), y.to(device))
    
    optimizer.zero_grad()
    # loss = y_pred.sum()
    # loss = loss.to(device)
    loss.backward()
    
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

predicted = model(x).detach().cpu().numpy()
plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'ro', label='Original data')
plt.plot(x.cpu().numpy(), predicted, label='Fitted line')
plt.legend()
plt.savefig('fitting_result.png')
plt.close()
