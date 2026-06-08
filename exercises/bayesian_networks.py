import numpy as np
import matplotlib.pyplot as plt

P_W = 0.002
P_U = 0.5
P_T = 0.06

P_S_given_W_U = 0.3
P_S_given_notW_U = 0.2
P_S_given_W_notU = 0.07
P_S_given_notW_notU = 0.02

P_A_given_T = 0.1
P_A_given_notT = 0.01

P_Z_given_S_A = 0.95
P_Z_given_notS_A = 0.90
P_Z_given_S_notA = 0.40
P_Z_given_notS_notA = 0.11


# Analytical version
P_S_analit = (P_S_given_W_U * P_W * P_U +
              P_S_given_notW_U * (1 - P_W) * P_U +
              P_S_given_W_notU * P_W * (1 - P_U) +
              P_S_given_notW_notU * (1 - P_W) * (1 - P_U))

P_S_given_U = P_S_given_W_U * P_W + P_S_given_notW_U * (1 - P_W)
P_A = P_A_given_T * P_T + P_A_given_notT * (1 - P_T)

P_Z_given_U = (P_Z_given_S_A * P_S_given_U * P_A +
               P_Z_given_notS_A * (1 - P_S_given_U) * P_A +
               P_Z_given_S_notA * P_S_given_U * (1 - P_A) +
               P_Z_given_notS_notA * (1 - P_S_given_U) * (1 - P_A))

P_UZ_analit = P_Z_given_U * P_U

P_Z_given_U_T = (P_Z_given_S_A * P_S_given_U * P_A_given_T +
                 P_Z_given_notS_A * (1 - P_S_given_U) * P_A_given_T +
                 P_Z_given_S_notA * P_S_given_U * (1 - P_A_given_T) +
                 P_Z_given_notS_notA * (1 - P_S_given_U) * (1 - P_A_given_T))

P_UZT_analit = P_Z_given_U_T * P_U * P_T

print("Results - analytical version")
print(f"P(S)       = {P_S_analit:.6f}")
print(f"P(UZ)      = {P_UZ_analit:.6f}")
print(f"P(UZ,T)    = {P_UZT_analit:.6f}\n")



# Monte Carlo Simulations
N = 500000

W = np.random.rand(N) < P_W
U = np.random.rand(N) < P_U
T = np.random.rand(N) < P_T

S = np.zeros(N, dtype=bool)
S[(W == True)  & (U == True)]  = np.random.rand(np.sum((W == True)  & (U == True)))  < P_S_given_W_U
S[(W == False) & (U == True)]  = np.random.rand(np.sum((W == False) & (U == True)))  < P_S_given_notW_U
S[(W == True)  & (U == False)] = np.random.rand(np.sum((W == True)  & (U == False))) < P_S_given_W_notU
S[(W == False) & (U == False)] = np.random.rand(np.sum((W == False) & (U == False))) < P_S_given_notW_notU

A = np.zeros(N, dtype=bool)
A[T == True]  = np.random.rand(np.sum(T == True))  < P_A_given_T
A[T == False] = np.random.rand(np.sum(T == False)) < P_A_given_notT

Z = np.zeros(N, dtype=bool)
Z[(S == True)  & (A == True)]  = np.random.rand(np.sum((S == True)  & (A == True)))  < P_Z_given_S_A
Z[(S == False) & (A == True)]  = np.random.rand(np.sum((S == False) & (A == True)))  < P_Z_given_notS_A
Z[(S == True)  & (A == False)] = np.random.rand(np.sum((S == True)  & (A == False))) < P_Z_given_S_notA
Z[(S == False) & (A == False)] = np.random.rand(np.sum((S == False) & (A == False))) < P_Z_given_notS_notA

print("Results - Monte Carlo Simulations")
print(f"P(S)       ~ {np.sum(S) / N:.6f}")
print(f"P(UZ)      ~ {np.sum(U & Z) / N:.6f}")
print(f"P(UZ,T)    ~ {np.sum(U & Z & T) / N:.6f}\n")


#convergence diagrams
steps = np.arange(500, N + 1, 500)

S_cumsum = np.cumsum(S)
P_S_time = S_cumsum[steps - 1] / steps

UZ_cumsum = np.cumsum(U & Z)
P_UZ_time = UZ_cumsum[steps - 1] / steps

UZT_cumsum = np.cumsum(U & Z & T)
P_UZT_time = UZT_cumsum[steps - 1] / steps

plt.figure(figsize=(12, 8))

# P(S) plot
plt.subplot(3, 1, 1)
plt.plot(steps, P_S_time, color='blue', label='Monte Carlo Simulation')
plt.axhline(y=P_S_analit, color='red', linestyle='--', label=f'Theory ({P_S_analit:.5f})')
plt.title('Monte Carlo method convergence diagram')
plt.ylabel('P(S)')
plt.legend()
plt.grid(True)

# P(UZ) plot
plt.subplot(3, 1, 2)
plt.plot(steps, P_UZ_time, color='green', label='Monte Carlo Simulation')
plt.axhline(y=P_UZ_analit, color='red', linestyle='--', label=f'Theory ({P_UZ_analit:.5f})')
plt.ylabel('P(UZ)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(steps, P_UZT_time, color='purple', label='Monte Carlo Simulation')
plt.axhline(y=P_UZT_analit, color='red', linestyle='--', label=f'Theory ({P_UZT_analit:.5f})')
plt.xlabel('Simulation number (n)')
plt.ylabel('P(UZ,T)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()