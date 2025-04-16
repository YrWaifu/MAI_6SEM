import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve  # Для нахождения точки пересечения

# Параметр
a = 4

# Система уравнений
def eq1(x1, x2):
    return x1**2 + x2**2 - a**2

def eq2(x1, x2):
    return x1 - np.exp(x2) + a

# Функция для решения системы уравнений
def system(vars):
    x1, x2 = vars
    return [eq1(x1, x2), eq2(x1, x2)]

# Начальное приближение (графически выбираем положительные значения)
initial_guess = [3, 2]

# Решение системы уравнений
intersection = fsolve(system, initial_guess)
x1_intersect, x2_intersect = intersection

# Ограничение области построения графика (только положительная сторона)
x1_vals = np.linspace(0, 5, 500)  # Только положительные значения x1
x2_vals = np.linspace(0, 5, 500)  # Только положительные значения x2
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Вычисляем значения функций
Z1 = eq1(X1, X2)
Z2 = eq2(X1, X2)

# Строим графики
plt.figure(figsize=(8, 6))
contour1 = plt.contour(X1, X2, Z1, levels=[0], colors='blue', label="$x_1^2 + x_2^2 - a^2 = 0$")
contour2 = plt.contour(X1, X2, Z2, levels=[0], colors='red', label="$x_1 - e^{x_2} + a = 0$")

# Добавление легенды
plt.plot([], [], color='blue', label="$x_1^2 + x_2^2 - a^2 = 0$")
plt.plot([], [], color='red', label="$x_1 - e^{x_2} + a = 0$")
plt.legend()

# Отмечаем точку пересечения
plt.scatter(x1_intersect, x2_intersect, color='green', zorder=5)
plt.text(x1_intersect, x2_intersect, f"({x1_intersect:.2f}, {x2_intersect:.2f})", fontsize=10, color="green")

# Оформление графика
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.title("Графическое решение системы уравнений (положительная область)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(0, 5)  # Ограничение оси x1
plt.ylim(0, 5)  # Ограничение оси x2
plt.show()

# Вывод координат точки пересечения
print(f"Точка пересечения: x1 ≈ {x1_intersect:.6f}, x2 ≈ {x2_intersect:.6f}")