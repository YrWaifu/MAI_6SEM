import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию
def f(x):
    return x**3 + x**2 - 2*x - 1

# Создаем массив значений x
x = np.linspace(-2, 2, 500)

# Вычисляем значения функции
y = f(x)

# Строим график
plt.plot(x, y, label="$f(x) = x^3 + x^2 - 2x - 1$")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.legend()
plt.title("График функции")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()