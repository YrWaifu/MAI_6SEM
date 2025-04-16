import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Параметры
a = 4
tolerance = 1e-6
max_iterations = 1000

# Система уравнений
def f1(x1, x2):
    return x1**2 + x2**2 - a**2

def f2(x1, x2):
    return x1 - np.exp(x2) + a

# Якобиан системы
def jacobian(x1, x2):
    return np.array([[2*x1, 2*x2],
                     [1, -np.exp(x2)]])

# Метод простой итерации
def simple_iteration(x1_0, x2_0, tolerance, max_iterations):
    x1, x2 = x1_0, x2_0
    errors = []
    for iteration in range(max_iterations):
        x1_new = np.sqrt(a**2 - x2**2)
        x2_new = np.log(x1 + a)
        error = np.sqrt((x1_new - x1)**2 + (x2_new - x2)**2)
        errors.append(error)
        if error < tolerance:
            break
        x1, x2 = x1_new, x2_new
    return x1, x2, errors

# Метод Ньютона
def newton_method(x1_0, x2_0, tolerance, max_iterations):
    x = np.array([x1_0, x2_0])
    errors = []
    for iteration in range(max_iterations):
        J = jacobian(x[0], x[1])
        F = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        delta_x = np.linalg.solve(J, -F)
        x_new = x + delta_x
        error = np.linalg.norm(delta_x)
        errors.append(error)
        if error < tolerance:
            break
        x = x_new
    return x[0], x[1], errors

# Графическое определение начального приближения
x1_vals = np.linspace(-5, 5, 400)
x2_vals = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
F1 = X1**2 + X2**2 - a**2
F2 = X1 - np.exp(X2) + a

plt.contour(X1, X2, F1, levels=[0], colors='r')
plt.contour(X1, X2, F2, levels=[0], colors='b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Графическое определение начального приближения')
plt.grid()
plt.show()

# Начальное приближение (по графику)
x1_0, x2_0 = 3, 1

# Решение методом простой итерации
x1_si, x2_si, errors_si = simple_iteration(x1_0, x2_0, tolerance, max_iterations)
print(f"Решение методом простой итерации: x1 = {x1_si}, x2 = {x2_si}")

# Решение методом Ньютона
x1_nm, x2_nm, errors_nm = newton_method(x1_0, x2_0, tolerance, max_iterations)
print(f"Решение методом Ньютона: x1 = {x1_nm}, x2 = {x2_nm}")

# Решение с использованием fsolve из scipy
def system(vars):
    x1, x2 = vars
    return [f1(x1, x2), f2(x1, x2)]

solution = fsolve(system, (x1_0, x2_0))
print(f"Решение с использованием fsolve: x1 = {solution[0]}, x2 = {solution[1]}")

# Анализ зависимости погрешности от количества итераций
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(errors_si, label='Simple Iteration')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Simple Iteration Error')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(errors_nm, label='Newton Method', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Newton Method Error')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
