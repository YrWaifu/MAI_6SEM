import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve  # Для решения системы встроенными методами

# Параметры
a = 4
tolerance = 1e-6
max_iterations = 100

# Система уравнений
def equations(vars):
    x1, x2 = vars
    eq1 = x1**2 + x2**2 - a**2
    eq2 = x1 - np.exp(x2) + a
    return [eq1, eq2]

# Метод простой итерации
def simple_iteration(phi1, phi2, initial_guess, tol, max_iter):
    x1, x2 = initial_guess
    errors = []
    for i in range(max_iter):
        x1_new = phi1(x1, x2)
        x2_new = phi2(x1, x2)
        error = np.sqrt((x1_new - x1)**2 + (x2_new - x2)**2)
        errors.append(error)
        if error < tol:
            return [x1_new, x2_new], errors
        x1, x2 = x1_new, x2_new
    return [x1, x2], errors

# Преобразования для метода простой итерации
def phi1(x1, x2):
    return np.sqrt(a**2 - x2**2)

def phi2(x1, x2):
    return np.log(x1 + a)

# Метод Ньютона
def newton_method(equations, jacobian, initial_guess, tol, max_iter):
    x = np.array(initial_guess, dtype=np.float64)
    errors = []
    for i in range(max_iter):
        f = np.array(equations(x), dtype=np.float64)
        J = np.array(jacobian(x), dtype=np.float64)
        delta = np.linalg.solve(J, -f)
        x += delta
        error = np.linalg.norm(delta)
        errors.append(error)
        if error < tol:
            return x.tolist(), errors
    return x.tolist(), errors

# Якобиан системы
def jacobian(vars):
    x1, x2 = vars
    return [
        [2*x1, 2*x2],
        [1, -np.exp(x2)]
    ]

# Начальное приближение (графически: положительные значения x1 и x2)
initial_guess = [3.5, 2]

# Решение методом простой итерации
root_simple, errors_simple = simple_iteration(phi1, phi2, initial_guess, tolerance, max_iterations)

# Решение методом Ньютона
root_newton, errors_newton = newton_method(equations, jacobian, initial_guess, tolerance, max_iterations)

# Решение встроенными методами (fsolve)
root_fsolve = fsolve(equations, initial_guess)

# Вывод результатов
print("Метод простой итерации:")
print(f"  Корень: x1 = {root_simple[0]:.6f}, x2 = {root_simple[1]:.6f}")
print(f"  Количество итераций: {len(errors_simple)}\n")

print("Метод Ньютона:")
print(f"  Корень: x1 = {root_newton[0]:.6f}, x2 = {root_newton[1]:.6f}")
print(f"  Количество итераций: {len(errors_newton)}\n")

print("Встроенный метод fsolve (scipy):")
print(f"  Корень: x1 = {root_fsolve[0]:.6f}, x2 = {root_fsolve[1]:.6f}\n")

# Анализ погрешности
# plt.figure(figsize=(10, 6))
# plt.plot(errors_simple, label="Метод простой итерации", marker='o')
# plt.plot(errors_newton, label="Метод Ньютона", marker='x')
# plt.yscale('log')
# plt.xlabel("Номер итерации")
# plt.ylabel("Погрешность")
# plt.title("Сравнение скорости сходимости методов")
# plt.legend()
# plt.grid()
# plt.show()