import torch
import torch.nn as nn
import numpy as np
import time

# Forçar o uso de Float64 (Double Precision) para gastar RAM e ganhar precisão
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

class BurgersPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def get_pde_loss(model, x, t, nu=0.01/np.pi):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual**2)

# --- CONFIGURAÇÃO DE ALTA CARGA ---
model = BurgersPINN()
# 30.000 pontos para testar a estabilidade dos 16GB
x_pde = (torch.rand(30000, 1) * 2 - 1)
t_pde = torch.rand(30000, 1)

optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, 
                              history_size=50, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = get_pde_loss(model, x_pde, t_pde)
    loss.backward()
    return loss

print(f"🚀 Iniciando Burgers PINN no ASUS (RAM Livre: ~11GB )")
start_time = time.time()

for i in range(10):
    loss = optimizer.step(closure)
    print(f"Época {i} | Loss: {loss.item():.10f}")

print(f"✅ Treino finalizado em: {time.time() - start_time:.2f} segundos")