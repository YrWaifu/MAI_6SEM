import numpy as np

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n), dtype=np.float64)
    U = A.astype(np.float64).copy()

    for i in range(n):
        for j in range(i + 1, n):  # Для всех строк ниже текущей
            L[j, i] = U[j, i] / U[i, i]  # Вычисляем множитель и записываем в матрицу L
            U[j, i:] -= L[j, i] * U[i, i:]  # Обновляем матрицу U

    np.fill_diagonal(L, 1)
    return L, U

def solve_lu(L, U, b):
    # Прямая подстановка (решаем Ly = b)
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Обратная подстановка (решаем Ux = y)
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def determinant(U):
    # Определитель верхней треугольной матрицы U равен произведению диагональных элементов
    return np.prod(np.diag(U))

def inverse_matrix(L, U):
    n = len(L)
    inv_A = np.zeros((n, n), dtype=np.float64)
    I = np.eye(n, dtype=np.float64)

    # Для каждого столбца единичной матрицы решаем систему Ax = I[:, i]
    for i in range(n):
        inv_A[:, i] = solve_lu(L, U, I[:, i])  # Решение записываем в соответствующий столбец обратной матрицы

    return inv_A

A = np.array([
    [-7, -9, 1, -9],
    [-6, -8, -5, 2],
    [-3, 6, 5, -9],
    [-2, 0, -5, -9]
], dtype=np.float64)
b = np.array([29, 42, 11, 75], dtype=np.float64)

L, U = lu_decomposition(A)

solution = solve_lu(L, U, b)
print("Решение системы (наше решение):", np.round(solution, 6))

numpy_solution = np.linalg.solve(A, b)
print("Решение системы (numpy.linalg.solve):", np.round(numpy_solution, 6))

print("Совпадают ли решения?", np.allclose(solution, numpy_solution))

det_A = determinant(U)
print("Определитель матрицы A:", np.round(det_A, 6))
print("Определитель матрицы A (numpy.linalg.det):", np.round(np.linalg.det(A), 6))
print("Совпадают ли определители?", np.isclose(det_A, np.linalg.det(A)))


inv_A = inverse_matrix(L, U)
print("Обратная матрица A:\n", np.round(inv_A, 6))

identity_check = np.dot(A, inv_A)
print("Проверка обратной матрицы (A * A^{-1}):\n", np.round(identity_check, 6))