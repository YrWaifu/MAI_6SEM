import numpy as np


# Реализовать алгоритм LU - разложения матриц (с выбором главного элемента) в виде программы.
# Используя разработанное программное обеспечение, решить систему линейных алгебраических уравнений (СЛАУ).
# Для матрицы СЛАУ вычислить определитель и обратную матрицу.

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.copy(A)
    P = np.eye(n)

    for i in range(n):
        # Выбор главного элемента
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if max_row != i:
            # Перестановка строк в U
            U[[i, max_row]] = U[[max_row, i]]
            # Перестановка строк в P
            P[[i, max_row]] = P[[max_row, i]]
            # Перестановка строк в L
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        # LU-разложение
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U


def solve_slu(P, L, U, b):
    n = len(b)
    # Решение Ly = Pb
    y = np.zeros(n)
    Pb = np.dot(P, b)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Решение Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def determinant(U):
    return np.prod(np.diag(U))


def inverse_matrix(P, L, U):
    n = len(L)
    inv_A = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        inv_A[:, i] = solve_slu(P, L, U, e)
    return inv_A


# Исходная система
A = np.array([[-7, -9, 1, -9],
              [-6, -8, -5, 2],
              [-3, 6, 5, -9],
              [-2, 0, -5, -9]], dtype=float)

b = np.array([29, 42, 11, 75], dtype=float)

# LU-разложение
P, L, U = lu_decomposition(A)

# Решение СЛАУ
x = solve_slu(P, L, U, b)
print("Решение СЛАУ:", x)

# Определитель матрицы A
det_A = determinant(U)
print("Определитель матрицы A:", det_A)

# Обратная матрица
inv_A = inverse_matrix(P, L, U)
print("Обратная матрица A^{-1}:")
print(inv_A)