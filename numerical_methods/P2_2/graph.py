import numpy as np
import matplotlib.pyplot as plt


a = 4

x1 = np.linspace(-a, a, 500)
x2 = np.linspace(-a, a, 500)
X1, X2 = np.meshgrid(x1, x2)

# Уравнения в виде функций
Z1 = X1**2 + X2**2 - a**2
Z2 = X1 - np.exp(X2) + a

# Построение графиков
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z1, levels=[0], colors='blue', label="$x_1^2 + x_2^2 - a^2 = 0$")
plt.contour(X1, X2, Z2, levels=[0], colors='red', label="$x_1 - e^{x_2} + a = 0$")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.title("Графическое решение системы")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(["$x_1^2 + x_2^2 - a^2 = 0$", "$x_1 - e^{x_2} + a = 0$"])
plt.show()