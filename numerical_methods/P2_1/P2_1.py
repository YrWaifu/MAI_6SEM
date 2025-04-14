import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 + x**2 - 2*x - 1

def df(x):
    return 3 * x**2 + 2 * x - 2

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

def phi(x):
    return (2 * x + 1 - x**2)**(1/3)

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
x0 = 1.5
tolerance = 1e-6
max_iterations = 100

# Решение
root_simple, errors_simple = simple_iteration(phi, x0, tolerance, max_iterations)
root_newton, errors_newton = newton_method(f, df, x0, tolerance, max_iterations)

# Вывод результатов
print("Метод простой итерации:")
print(f"  Корень: x = {root_simple:.6f}")
print(f"  Количество итераций: {len(errors_simple)}\n")

print("Метод Ньютона:")
print(f"  Корень: x = {root_newton:.6f}")
print(f"  Количество итераций: {len(errors_newton)}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(errors_simple, label="Метод простой итерации", marker='o')
plt.plot(errors_newton, label="Метод Ньютона", marker='x')
plt.yscale('log')
plt.xlabel("Номер итерации")
plt.ylabel("Погрешность")
plt.title("Сравнение скорости сходимости методов")
plt.legend()
plt.grid()
plt.show()