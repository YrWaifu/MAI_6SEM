import numpy as np

def qr_algorithm(A, tol=1e-9, max_iterations=1000):
    n = A.shape[0]
    Ak = A.copy()

    for iteration in range(max_iterations):
        # Выполняем QR-разложение
        Q, R = np.linalg.qr(Ak)
        # Обновляем матрицу Ak
        Ak = R @ Q

        # Проверяем условие сходимости
        off_diagonal_norm = np.sqrt(np.sum(np.tril(Ak, -1)**2))
        if off_diagonal_norm < tol:
            break

    # Собственные значения находятся на диагонали матрицы Ak
    eigenvalues = np.diag(Ak)
    return eigenvalues

# Пример использования
A = np.array([
    [-5, -8, 4],
    [4, 2, 6],
    [-2, 5, -6]
], dtype=np.float64)

# Собственные значения с использованием QR-алгоритма
eigenvalues_qr = qr_algorithm(A)
print("Собственные значения (QR-алгоритм):", eigenvalues_qr)

# Собственные значения с использованием NumPy
eigenvalues_numpy, _ = np.linalg.eig(A)
print("Собственные значения (NumPy):", eigenvalues_numpy)

# Проверка корректности
print("Совпадают ли собственные значения?", np.allclose(np.sort(eigenvalues_qr), np.sort(eigenvalues_numpy)))