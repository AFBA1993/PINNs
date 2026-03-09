import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os

"""
NOTAS DE PESQUISA - SESSÃO 01: PINNs & EQUAÇÃO DE BURGERS (8h DE ESTUDO)
-----------------------------------------------------------------------
ESTADO ATUAL:
- Arquitetura: MLP 3-4 camadas (60-80 neurônios), ativação Tanh.
- Otimização: Adam (1e-3 a 5e-4), 3000-5000 épocas.
- Regime Físico: Testada transição de Nu (0.3183 -> 0.000003).

PRINCIPAIS APRENDIZADOS:
1. SENSIBILIDADE DO NU: Identificado que Nu < 0.01 atua como 'botão de 
   dificuldade'. O erro residual concentra-se na frente de choque (x > 0.8).
2. LIMITAÇÃO DA MALHA UNIFORME: Pontos de colocação igualmente espaçados
   falham em capturar gradientes íngremes. Resíduo estagna em ~10^-1.
3. DIAGNÓSTICO VISUAL: O 'Mapa de Calor do Resíduo' foi validado como a 
   melhor ferramenta para identificar falhas na satisfação da PDE.
4. CONSERVAÇÃO: Iniciada análise de conservação de massa (integral de u).

PRÓXIMA SESSÃO: 'SHOCK CAPTURING & PRECISION 10^-7'
- Estratégia: Residual-based Adaptive Refinement (RAR).
- Métrica: Monitoramento rigoroso da integral de conservação.
- Meta: Reduzir a 'mancha amarela' de erro local no choque.
-----------------------------------------------------------------------
"""




# Configuração para rodar via SSH sem interface gráfica
import matplotlib
matplotlib.use('Agg') 

# ---------------------------------------------------------
# 1. Configuração de Dispositivo e Arquitetura
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 1)
        )
    def forward(self, x):
        return self.net(x)

model = TinyNN().to(device)

# ---------------------------------------------------------
# 2. Física e Setup (Nu variável)
# ---------------------------------------------------------
nu_val = 0.00001 / np.pi 
nu = torch.tensor([nu_val], device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 3000

# Pontos de Colocação
x = torch.linspace(0, 1, 80); t = torch.linspace(0, 1, 80)
X, T = torch.meshgrid(x, t, indexing="ij")
XT = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1).to(device).requires_grad_(True)

x_ic = torch.linspace(0, 1, 150).reshape(-1, 1).to(device)
u_exact_ic = torch.sin(torch.pi * x_ic)
t_bc = torch.linspace(0, 1, 150).reshape(-1, 1).to(device)

history = {'pde': [], 'ic': [], 'total': []}

# ---------------------------------------------------------
# 3. Treinamento (Prints no terminal SSH)
# ---------------------------------------------------------
print(f"\n--- Treinando no Servidor | nu = {nu_val:.6f} ---")

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    u = model(XT)
    grad_u = torch.autograd.grad(u, XT, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, XT, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    loss_pde = torch.mean((u_t + u * u_x - nu * u_xx)**2)
    u_ic_pred = model(torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1))
    loss_ic = torch.mean((u_ic_pred - u_exact_ic)**2)
    loss_bc = torch.mean(model(torch.cat([torch.zeros_like(t_bc), t_bc], dim=1))**2) + \
              torch.mean(model(torch.cat([torch.ones_like(t_bc), t_bc], dim=1))**2)
    
    loss = loss_pde + (loss_ic * 2.0) + loss_bc
    loss.backward()
    optimizer.step()
    
    history['pde'].append(loss_pde.item())
    history['ic'].append(loss_ic.item())
    history['total'].append(loss.item())
    
    print(f"Iter {epoch:4d} | Loss: {loss.item():.6e} | PDE: {loss_pde.item():.6e} | IC: {loss_ic.item():.6e}")

# ---------------------------------------------------------
# 4. Geração e Salvamento da Imagem
# ---------------------------------------------------------
model.eval()
print("\n--- Gerando arquivo de imagem ---")
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
plt.suptitle(f"Resultados PINN (SSH) - Nu = {nu_val:.6f}", fontsize=16)

# [Plot 1: Convergência]
axs[0, 0].semilogy(history['total'], label='Total')
axs[0, 0].semilogy(history['pde'], label='PDE')
axs[0, 0].set_title("Perdas Log")
axs[0, 0].legend()

# [Plot 2: Amplitude]
times_check = np.linspace(0, 1, 50)
max_vals = [model(torch.cat([torch.linspace(0,1,100).reshape(-1,1).to(device), torch.full((100,1), t_v).to(device)], dim=1)).detach().cpu().max().item() for t_v in times_check]
axs[0, 1].plot(times_check, max_vals, 'r')
axs[0, 1].set_title("Decaimento Amplitude")

# [Plot 3: Perfis]
for t_v in [0.0, 0.5, 1.0]:
    u_p = model(torch.cat([torch.linspace(0,1,100).reshape(-1,1).to(device), torch.full((100,1), t_v).to(device)], dim=1)).detach().cpu()
    axs[0, 2].plot(np.linspace(0,1,100), u_p.numpy(), label=f"t={t_v}")
axs[0, 2].legend(); axs[0, 2].set_title("Perfis Temporais")

# [Plot 4: Resíduo]
res_map = (u_t + u * u_x - nu * u_xx).detach().cpu().reshape(80, 80)
im = axs[1, 0].contourf(X.numpy(), T.numpy(), np.abs(res_map.numpy()), levels=50, cmap='inferno')
fig.colorbar(im, ax=axs[1, 0]); axs[1, 0].set_title("Mapa de Resíduo")

# [Plot 5: Gradiente]
x_s = torch.linspace(0,1,400).reshape(-1,1).to(device); x_s.requires_grad_(True)
u_s = model(torch.cat([x_s, torch.full_like(x_s, 0.5)], dim=1))
u_x_s = torch.autograd.grad(u_s, x_s, grad_outputs=torch.ones_like(u_s))[0].detach().cpu()
axs[1, 1].plot(np.linspace(0,1,400), u_x_s.numpy(), 'purple')
axs[1, 1].set_title("Gradiente u_x (t=0.5)")

# [Plot 6: Solução Final]
im6 = axs[1, 2].imshow(u.detach().cpu().reshape(80,80).T.numpy(), extent=[0,1,0,1], origin='lower', aspect='auto')
fig.colorbar(im6, ax=axs[1, 2]); axs[1, 2].set_title("Campo u(x,t)")

# SALVANDO O ARQUIVO
filename = f"burgers_nu_{nu_val:.5f}.png"
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(filename, dpi=150)
print(f"--- Sucesso! Imagem salva como: {filename} ---")
