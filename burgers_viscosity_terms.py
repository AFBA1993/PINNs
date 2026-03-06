import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------
# Device
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------
# Neural Network
# -----------------
class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),  # input: (x, t)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)   # output: u(x,t)
        )

    def forward(self, x):
        return self.net(x)

model = TinyNN().to(device)

# -----------------
# Collocation Points (Interior)
# -----------------
x = torch.linspace(0, 1, 50)
t = torch.linspace(0, 1, 50)

X, T = torch.meshgrid(x, t, indexing="ij")

X = X.reshape(-1, 1)
T = T.reshape(-1, 1)

XT = torch.cat([X, T], dim=1).to(device)
XT.requires_grad_(True)

# -----------------
# Initial Condition (t = 0)
# u(x,0) = sin(pi x)
# -----------------
x_ic = torch.linspace(0, 1, 50).reshape(-1, 1)
t_ic = torch.zeros_like(x_ic)

XT_ic = torch.cat([x_ic, t_ic], dim=1).to(device)
u_exact_ic = torch.sin(torch.pi * x_ic).to(device)

# -----------------
# Boundary Conditions
# u(0,t)=0 , u(1,t)=0
# -----------------
t_bc = torch.linspace(0, 1, 50).reshape(-1, 1)

x0 = torch.zeros_like(t_bc)
x1 = torch.ones_like(t_bc)

XT_bc0 = torch.cat([x0, t_bc], dim=1).to(device)
XT_bc1 = torch.cat([x1, t_bc], dim=1).to(device)

# -----------------
# Optimizer
# -----------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nu = 0.01
epochs = 3000

# -----------------
# Training Loop
# -----------------
for epoch in range(epochs):

    optimizer.zero_grad()

    # ---- PDE (Interior) ----
    u = model(XT)

    grad_u = torch.autograd.grad(
        u,
        XT,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    
    grad_2u = torch.autograd.grad(
        u_x,
        XT,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    u_xx = grad_2u[:,0:1]   
    
    



    residual = u_t - nu * u_xx + u * u_x
    loss_pde = torch.mean(residual ** 2)

    # ---- Initial Condition ----
    u_ic = model(XT_ic)
    loss_ic = torch.mean((u_ic - u_exact_ic) ** 2)

    # ---- Boundary Conditions ----
    u_bc0 = model(XT_bc0)
    u_bc1 = model(XT_bc1)
    loss_bc = torch.mean(u_bc0 ** 2) + torch.mean(u_bc1 ** 2)

    # ---- Total Loss ----
    loss = loss_pde + loss_ic + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss_PDE: {loss_pde.item():.6e} | Loss_IC: {loss_ic.item():.6e} | Loss_BC: {loss_bc.item():.6e}" )

# -----------------
# Evaluation Grid
# -----------------
x_test = torch.linspace(0, 1, 100)
t_test = torch.linspace(0, 1, 100)

X_test, T_test = torch.meshgrid(x_test, t_test, indexing="ij")

XT_test = torch.cat(
    [X_test.reshape(-1, 1), T_test.reshape(-1, 1)],
    dim=1
).to(device)

u_pred = model(XT_test).detach().cpu()
U_pred = u_pred.reshape(100, 100)


# -----------------
# Amplitude Tracking Over Time
# -----------------
times_to_check = torch.linspace(0, 1, 20)
min_vals = []
max_vals = []

for t_val in times_to_check:
    x_line = torch.linspace(0, 1, 200).reshape(-1, 1)
    t_line = torch.full_like(x_line, t_val)

    XT_line = torch.cat([x_line, t_line], dim=1).to(device)
    u_line = model(XT_line).detach().cpu()

    min_vals.append(u_line.min().item())
    max_vals.append(u_line.max().item())

plt.figure()
plt.plot(times_to_check, min_vals, label="min(u)")
plt.plot(times_to_check, max_vals, label="max(u)")
plt.legend()
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("Amplitude Evolution Over Time")
plt.show()


# -----------------
# Gradient Magnitude at t = 0.5
# -----------------
x_line = torch.linspace(0, 1, 400).reshape(-1, 1)
t_line = torch.full_like(x_line, 0.5)

XT_line = torch.cat([x_line, t_line], dim=1).to(device)
XT_line.requires_grad_(True)

u_line = model(XT_line)

grad_u = torch.autograd.grad(
    u_line,
    XT_line,
    grad_outputs=torch.ones_like(u_line),
    create_graph=False
)[0]

u_x_line = grad_u[:, 0].detach().cpu()

plt.figure()
plt.plot(x_line.detach().cpu(), u_x_line)
plt.xlabel("x")
plt.ylabel("u_x")
plt.title("Spatial Gradient at t=0.5")
plt.show()

print("Max |u_x|:", u_x_line.abs().max().item())


# -----------------
# Burgers PINN Diagnostic Dashboard
# -----------------
import matplotlib.pyplot as plt

# Evaluate on the same grid used for testing
XT_test.requires_grad_(True)
u_pred = model(XT_test)

# Compute gradients
grad_u = torch.autograd.grad(
    u_pred,
    XT_test,
    grad_outputs=torch.ones_like(u_pred),
    create_graph=True
)[0]

u_x = grad_u[:,0:1]
u_t = grad_u[:,1:2]

# Second derivative (viscous term)
grad_2u = torch.autograd.grad(
    u_x,
    XT_test,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True
)[0]
u_xx = grad_2u[:,0:1]

# PDE residual
residual = u_t - nu * u_xx + u_pred * u_x

# Reshape for plotting
U_pred = u_pred.detach().cpu().reshape(100,100)
U_x = u_x.detach().cpu().reshape(100,100)
Res = residual.detach().cpu().reshape(100,100)

# ---- Plot 1: u(x,t=0.5) ----
plt.figure(figsize=(6,4))
plt.plot(X_test[:100,0].cpu(), U_pred[:,51])
plt.xlabel("x")
plt.ylabel("u(x, t=0.5)")
plt.title("Burgers PINN: Wave Profile at t=0.5")
plt.grid(True)
plt.show()

# ---- Plot 2: Spatial Gradient u_x(x,t=0.5) ----
plt.figure(figsize=(6,4))
plt.plot(X_test[:100,0].cpu(), U_x[:,51])
plt.xlabel("x")
plt.ylabel("u_x(x, t=0.5)")
plt.title("Burgers PINN: Spatial Gradient at t=0.5")
plt.grid(True)
plt.show()

# Create proper meshgrid for plotting
x_plot = torch.linspace(0, 1, 100)
t_plot = torch.linspace(0, 1, 100)
X_plot, T_plot = torch.meshgrid(x_plot, t_plot, indexing="ij")

# Reshape residual
Res_grid = residual.detach().cpu().reshape(100, 100)

# Plot residual heatmap
plt.figure(figsize=(6,5))
plt.contourf(X_plot, T_plot, Res_grid, levels=50, cmap="viridis")
plt.colorbar(label="PDE Residual")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Burgers PINN: PDE Residual Map")
plt.show()

# ---- Diagnostics ----
print("Max u:", U_pred.max().item())
print("Min u:", U_pred.min().item())
print("Max |u_x|:", U_x.abs().max().item())
print("Max residual:", Res.abs().max().item())
print("Min residual:", Res.min().item())

# -----------------
# Visualization of u(x,t) at multiple times
# -----------------
times_to_plot = [0.25, 0.5, 0.75, 1.0]
plt.figure(figsize=(8,5))

for t_val in times_to_plot:
    # Find index closest to this time in t_test
    t_idx = (torch.abs(t_test - t_val)).argmin().item()
    plt.plot(X_test[:,0].cpu(), U_pred[:, t_idx].cpu(), label=f"t = {t_val}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"Burgers Solution (nu={nu})")
plt.legend()
plt.grid(True)
plt.show()

# -----------------
# Plot gradient u_x(x,t=0.5)
# -----------------
XT_test.requires_grad_(True)
u_test = model(XT_test)
grad_u_test = torch.autograd.grad(
    u_test,
    XT_test,
    grad_outputs=torch.ones_like(u_test),
    create_graph=False
)[0]

u_x_test = grad_u_test[:,0].detach().cpu().reshape(100,100)

plt.figure(figsize=(8,5))
plt.plot(X_test[:,0].cpu(), u_x_test[:, t_idx].cpu())
plt.xlabel("x")
plt.ylabel("u_x(x,t=0.5)")
plt.title(f"Spatial Gradient at t=0.5 (nu={nu})")
plt.grid(True)
plt.show()

# -----------------
# Optional: Residual map (u_t - nu*u_xx + u*u_x)
# -----------------
grad_2u_test = torch.autograd.grad(
    grad_u_test[:,0:1],
    XT_test,
    grad_outputs=torch.ones_like(grad_u_test[:,0:1]),
    create_graph=False
)[0][:,0:1]

u_xx_test = grad_2u_test.detach().cpu().reshape(100,100)
u_t_test = grad_u_test[:,1].detach().cpu().reshape(100,100)

# Corrected Burgers Residual:
residual_test = u_t_test - nu*u_xx_test + U_pred * u_x_test  # U_pred already (u)

plt.figure(figsize=(8,5))
plt.contourf(X_test, T_test, residual_test, levels=50)
plt.colorbar(label="Residual")
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Residual Map")
plt.show()




# ---------------------------
# Burgers Equation with Viscosity (nu > 0)
# ---------------------------
# Adding the viscosity term nu * u_xx to the PDE: 
#     u_t + u * u_x = nu * u_xx
#
# Effect:
# - nu > 0 introduces diffusion, smoothing out steep gradients.
# - Small nu (near 0) → inviscid behavior, steep gradients form, almost like shock waves.
# - Large nu → solution is smoothed, gradients are smaller, amplitudes decay more slowly.
#
# What to observe in diagnostics:
# - u(x,t) over time:
#     * Check maximum and minimum values.
#     * Steepness of u_x: should decrease as nu increases.
# - Residual map:
#     * Values should remain small (loss_pde) if PINN is learning correctly.
#     * Sharp gradients in u may cause larger residuals if nu is too small.
# - Loss components:
#     * loss_pde, loss_ic, loss_bc should be balanced.
#     * For small nu, you may need higher resolution or adaptive sampling to capture steep gradients.
# - Intermediate plots at t = 0.25, 0.5, 0.75:
#     * With small nu, expect steep slopes (near shock formation) in u(x,t).
#     * With large nu, u(x,t) is smoother.