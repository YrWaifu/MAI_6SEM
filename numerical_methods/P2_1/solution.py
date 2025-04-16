import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve  # Для решения уравнения встроенными методами

# Определяем функцию
def f(x):
    return x**3 + x**2 - 2*x - 1

# Производная функции
def df(x):
    return 3 * x**2 + 2 * x - 2

# Метод простой итерации
def simple_iteration(phi, x0, tol, max_iter):
    x = x0
    errors = []
    for i in range(max_iter):
        x_new = phi(x)
        error = abs(x_new - x)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors

# Преобразование для метода простой итерации
def phi(x):
    return (2 * x + 1 - x**2)**(1/3)

# Метод Ньютона
def newton_method(f, df, x0, tol, max_iter):
    x = x0
    errors = []
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Производная равна нулю. Метод Ньютона не применим.")
        x_new = x - fx / dfx
        error = abs(x_new - x)
        errors.append(error)
        if error < tol:
            return x_new, errors
        x = x_new
    return x, errors

# Параметры
x0 = 1.25
tolerance = 1e-6
max_iterations = 100

# Решение методом простой итерации
root_simple, errors_simple = simple_iteration(phi, x0, tolerance, max_iterations)

# Решение методом Ньютона
root_newton, errors_newton = newton_method(f, df, x0, tolerance, max_iterations)

# Решение встроенными методами (fsolve)
root_fsolve = fsolve(f, x0)[0]  # fsolve возвращает массив, берем первый элемент

# Вывод результатов
print("Метод простой итерации:")
print(f"  Корень: x = {root_simple:.6f}")
print(f"  Количество итераций: {len(errors_simple)}\n")

print("Метод Ньютона:")
print(f"  Корень: x = {root_newton:.6f}")
print(f"  Количество итераций: {len(errors_newton)}\n")

print("Встроенный метод fsolve (scipy):")
print(f"  Корень: x = {root_fsolve:.6f}\n")

# Визуализация
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