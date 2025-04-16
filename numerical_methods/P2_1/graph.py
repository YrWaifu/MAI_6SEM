import numpy as np
import matplotlib.pyplot as plt

# Определяем функции
def g(x):
    return x**3 + x**2

def h(x):
    return 2*x + 1

# Задаем промежуток [0, 1]
x_vals = np.linspace(0, 2, 500)

# Вычисляем значения функций
g_vals = g(x_vals)
h_vals = h(x_vals)

# Строим графики
plt.figure(figsize=(8, 6))
plt.plot(x_vals, g_vals, label="$g(x) = x^3 + x^2$", color="blue")
plt.plot(x_vals, h_vals, label="$h(x) = 2x + 1$", color="red")

# Находим точку пересечения (приближенно)
intersection_x = None
for i in range(len(x_vals) - 1):
    if (g_vals[i] - h_vals[i]) * (g_vals[i + 1] - h_vals[i + 1]) < 0:  # Пересечение между i и i+1
        intersection_x = (x_vals[i] + x_vals[i + 1]) / 2  # Приближенное значение x
        break

# Отмечаем точку пересечения на графике
if intersection_x is not None:
    intersection_y = g(intersection_x)  # Или h(intersection_x), так как они равны в точке пересечения
    plt.scatter(intersection_x, intersection_y, color="green", zorder=5)
    plt.text(intersection_x, intersection_y, f"({intersection_x:.3f}, {intersection_y:.3f})", fontsize=10, color="green")

# Добавляем легенду и оформление
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.legend()
plt.title("Графическое решение уравнения $f(x) = 0$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

# Вывод приближенного начального значения
if intersection_x is not None:
    print(f"Начальное приближение (точка пересечения): x ≈ {intersection_x:.6f}")