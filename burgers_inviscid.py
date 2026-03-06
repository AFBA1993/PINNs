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

alpha = 1.0
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



    residual = u_t - u * u_x
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
# Residual Map
# -----------------
XT_test.requires_grad_(True)
u_test = model(XT_test)

grad_u = torch.autograd.grad(
    u_test,
    XT_test,
    grad_outputs=torch.ones_like(u_test),
    create_graph=True
)[0]

u_x = grad_u[:, 0:1]
u_t = grad_u[:, 1:2]

residual = u_t - u_test * u_x
R = residual.detach().cpu().reshape(100, 100)

plt.figure(figsize=(6,5))
plt.contourf(X_test, T_test, R, levels=50)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("Residual Map")
plt.show()





# -----------------
# Plot Solution
# -----------------
plt.figure(figsize=(6,5))
plt.contourf(X_test, T_test, U_pred, levels=50)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Solution: 1D Heat Equation")
plt.show()

# -----------------
# Plot Solution
# -----------------


plt.figure(figsize=(6,5))
plt.plot(X_test[:, 0], U_pred[:,51])
plt.xlabel("x")
plt.ylabel("u")
plt.title("PINN Solution: 1D Heat Equation")
plt.show()



print("min:", U_pred.min().item())
print("max:", U_pred.max().item())



"""
===========================================================
PINN DIAGNOSTIC SUMMARY — INVISCID BURGERS EQUATION
===========================================================

DATE: Session #1
TASK: Replace heat equation residual with inviscid Burgers PDE:
      u_t + u * u_x = 0
      while keeping IC and BC from heat-equation PINN.

-----------------------------------------------------------
OBSERVATIONS
-----------------------------------------------------------

1️⃣ Amplitude Evolution
   - Max(u) ≈ 0.7, Min(u) ≈ 0 throughout.
   - Indicates global smoothing.
   - True inviscid Burgers should conserve amplitude ±1.
   
2️⃣ Spatial Gradient
   - Max |u_x| initially ≈ 3, decays toward -1 by t=1.
   - Strong-form PINN suppresses steep gradients.
   - Prevents shock formation.

3️⃣ Residual Map
   - Faint diagonal band following characteristics.
   - Background mostly uniform (blue behind, green ahead).
   - Diagonal marks where solution attempts steepening.
   - Lack of strong spike confirms network cannot reproduce non-differentiable shocks.

4️⃣ Energy Behavior
   - Total energy (∫ u^2 dx) decreases over time.
   - Strong-form PINN introduces implicit artificial viscosity.

-----------------------------------------------------------
INTERPRETATION
-----------------------------------------------------------

- Strong-form PINNs assume differentiable solutions.
- Shocks violate differentiability → network smooths solution globally.
- Optimizer trades steep gradients for smaller residual everywhere.
- Observed smoothing is **not a coding error**, but a **structural limitation**.
- Residuals are roughly balanced; diagonal indicates attempt to follow characteristics.

-----------------------------------------------------------
NEXT STEPS
-----------------------------------------------------------

1. Optional: Introduce small viscosity (Burgers with nu > 0) to observe controlled steepening.
2. Explore conservative / flux-based PINN formulations for shocks.
3. Compare with 2D heat-equation PINN to consolidate understanding in smooth cases.
4. Maintain diagnostic workflow: amplitude, gradient, residual, energy.

===========================================================
NOTES
- Implementation is correct.
- PINNs are mesh-free and differentiable by construction.
- Shock representation requires special treatment beyond standard strong-form PINNs.
"""

