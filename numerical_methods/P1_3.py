import numpy as np

# Функция для проверки диагонального преобладания
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        sum_off_diagonal = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) <= sum_off_diagonal:
            return False
    return True

# Метод простых итераций
def jacobi_method(A, b, x0, tol, max_iterations):
    if not is_diagonally_dominant(A):
        raise ValueError("Матрица A не является диагонально доминирующей. Метод Якоби может не сойтись.")

    n = len(b)  # Размерность системы
    x = x0.copy()  # Начальное приближение

    for k in range(max_iterations):
        x_new = np.zeros_like(x)  # Новое приближение
        for i in range(n):
            # Вычисляем сумму, исключая диагональный элемент
            # (произведение матрицы А и текущего вектора х)
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            # Вычисляем новое значение x[i]
            x_new[i] = (b[i] - s) / A[i][i]

        # Проверяем условие выхода
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new

    return x, max_iterations

# Метод Зейделя (метод Гаусса-Зейделя)
def gauss_seidel_method(A, b, x0, tol, max_iterations):
    if not is_diagonally_dominant(A):
        raise ValueError("Матрица A не является диагонально доминирующей. Метод Гаусса-Зейделя может не сойтись.")

    n = len(b)  # Размерность системы
    x = x0.copy()  # Начальное приближение

    for k in range(max_iterations):
        x_old = x.copy()  # Сохраняем предыдущее приближение
        for i in range(n):
            # Вычисляем сумму проивзедений для уже обновленных значений
            s1 = sum(A[i][j] * x[j] for j in range(i))
            # Вычисляем сумму произведений элементов матрицы А на "старые" значения х
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            # Вычисляем новое значение x[i]
            x[i] = (b[i] - s1 - s2) / A[i][i]

        # Проверяем условие выхода
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1

    return x, max_iterations

# Определяем систему уравнений
A = np.array([
    [12, -3, -1, 3],
    [5, 20, 9, 1],
    [6, -3, -21, -7],
    [8, -7, 3, -27]
])
b = np.array([-31, 90, 119, 71])
x0 = np.zeros(len(b))
tol = 1e-9
max_iterations = 1000

print("Eps: ", tol)
jacobi_solution, jacobi_iterations = jacobi_method(A, b, x0, tol, max_iterations)
print("Решение методом простых итераций:",  np.round(jacobi_solution, 6))
print("Количество итераций этим методом:", jacobi_iterations)

gs_solution, gs_iterations = gauss_seidel_method(A, b, x0, tol, max_iterations)
print("Решение методом Гаусса-Зейделя:", np.round(gs_solution, 6))
print("Количество итераций методом Гаусса-Зейделя:", gs_iterations)
