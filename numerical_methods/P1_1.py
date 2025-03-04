import numpy as np


# Реализовать алгоритм LU - разложения матриц (с выбором главного элемента) в виде программы.
# Используя разработанное программное обеспечение, решить систему линейных алгебраических уравнений (СЛАУ).
# Для матрицы СЛАУ вычислить определитель и обратную матрицу.

def lu_decomposition_with_pivoting(A):
    n = len(A)
    L = np.zeros((n, n), dtype=np.float64)
    U = A.astype(np.float64).copy()  # Ensure U is float64
    P = np.eye(n, dtype=np.float64)

    for i in range(n):
        # Pivoting
        max_row = np.argmax(abs(U[i:, i])) + i
        if i != max_row:
            U[[i, max_row]] = U[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        # LU Decomposition
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    np.fill_diagonal(L, 1)
    return P, L, U


def solve_lu(P, L, U, b):
    # Forward substitution to solve Ly = Pb
    Pb = np.dot(P, b)
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Backward substitution to solve Ux = y
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def determinant(U, P):
    det = np.prod(np.diag(U))
    det *= (-1) ** np.sum(P != np.eye(len(P)))
    return det


def inverse_matrix(P, L, U):
    n = len(L)
    inv_A = np.zeros((n, n), dtype=np.float64)
    I = np.eye(n, dtype=np.float64)

    for i in range(n):
        inv_A[:, i] = solve_lu(P, L, U, I[:, i])

    return inv_A


# Define the system of equations
A = np.array([
    [-7, -9, 1, -9],
    [-6, -8, -5, 2],
    [-3, 6, 5, -9],
    [-2, 0, -5, -9]
])
b = np.array([29, 42, 11, 75], dtype=np.float64)

# Perform LU decomposition with pivoting
P, L, U = lu_decomposition_with_pivoting(A)

# Solve the system
solution = solve_lu(P, L, U, b)
print("Solution:", solution)

# Compute the determinant
det_A = determinant(U, P)
print("Determinant:", det_A)

# Compute the inverse of the matrix
inv_A = inverse_matrix(P, L, U)
print("Inverse Matrix:\n", inv_A)