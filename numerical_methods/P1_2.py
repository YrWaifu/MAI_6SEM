import numpy as np

def parse_tridiagonal_matrix(A):
    n = len(A)
    a = np.zeros(n)  # Поддиагональные элементы
    b = np.zeros(n)  # Диагональные элементы
    c = np.zeros(n)  # Наддиагональные элементы

    for i in range(n):
        if i > 0:
            a[i] = A[i, i - 1]
        b[i] = A[i, i]
        if i < n - 1:
            c[i] = A[i, i + 1]

    return a, b, c

def thomas_algorithm(a, b, c, d):
    n = len(d)

    P = [0] * n  # Вспомогательный массив для хранения модифицированных коэффициентов P
    Q = [0] * n  # Вспомогательный массив для хранения модифицированных правых частей Q

    P[0] = c[0] / b[0]
    Q[0] = d[0] / b[0]

    # Прямой ход
    for i in range(1, n):
        temp = b[i] - a[i] * P[i - 1]  # Вычисление временного значения
        P[i] = c[i] / temp if i < n - 1 else 0  # Модификация коэффициента P
        Q[i] = (d[i] - a[i] * Q[i - 1]) / temp  # Модификация правой части Q

    print("Прогоночные коэффициенты P:", np.round(P, 6))
    print("Прогоночные коэффициенты Q:", np.round(Q, 6))

    # Обратный ход
    x = [0] * n  # Массив для хранения решения
    x[-1] = Q[-1]  # Инициализация последнего элемента решения
    for i in range(n - 2, -1, -1):
        x[i] = Q[i] - P[i] * x[i + 1]  # Вычисление текущего элемента решения

    return x  # Возвращаем решение системы

A = np.array([
    [8, -4, 0, 0, 0],
    [-2, 12, -7, 0, 0],
    [0, 2, -9, 1, 0],
    [0, 0, -8, 17, -4],
    [0, 0, 0, -7, 13]
], dtype=np.float64)
d = np.array([32, 15, -10, 133, -76], dtype=np.float64)

a, b, c = parse_tridiagonal_matrix(A)

solution = thomas_algorithm(a, b, c, d)
print("Решение системы (метод прогонки):", np.round(solution, 6))

numpy_solution = np.linalg.solve(A, d)
print("Решение системы (numpy.linalg.solve):", np.round(numpy_solution, 6))

print("Совпадают ли решения?", np.allclose(solution, numpy_solution))
