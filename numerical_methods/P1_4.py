import numpy as np

# Функция для отображения матрицы
def show_matrix(mtx):
    for row in mtx:
        print(" ".join(f"{elem:15.10f}" for elem in row))

# Функция для транспонирования матрицы
def transpose_matrix(mtx):
    return np.transpose(mtx)

# Функция для умножения двух матриц
def multiply_matrices(first, second):
    if first.shape[1] != second.shape[0]:
        raise ValueError("Матрицы невозможно перемножить!")
    return np.dot(first, second)

# Функция для получения индексов максимального недиагонального элемента
def get_indices(mtx):
    n = mtx.shape[0]
    max_val = 0
    indices = (0, 1)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(mtx[i, j]) > max_val:
                max_val = abs(mtx[i, j])
                indices = (i, j)
    return indices

# Функция для получения матрицы вращения H
def get_h_matrix(mtx, indices):
    f_ind, s_ind = indices
    rotation_angle = 0.5 * np.arctan2(2 * mtx[f_ind, s_ind], mtx[f_ind, f_ind] - mtx[s_ind, s_ind])
    H = np.eye(mtx.shape[0])
    H[f_ind, f_ind] = H[s_ind, s_ind] = np.cos(rotation_angle)
    H[f_ind, s_ind] = -np.sin(rotation_angle)
    H[s_ind, f_ind] = np.sin(rotation_angle)
    return H

# Функция для вычисления ошибки
def calculate_the_error(mtx):
    error = 0
    n = mtx.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            error += mtx[i, j]**2
    return np.sqrt(error)

# Функция для нахождения собственных значений и векторов
def get_eigens(mtx, epsilon=1e-9):
    A_matrix = mtx.copy()
    H_matrices = []
    stop = False
    while not stop:
        indices = get_indices(A_matrix)
        H = get_h_matrix(A_matrix, indices)
        H_matrices.append(H)
        new_A_matrix = multiply_matrices(transpose_matrix(H), A_matrix)
        A_matrix = multiply_matrices(new_A_matrix, H)
        if calculate_the_error(A_matrix) < epsilon:
            stop = True

    eigenvalues = np.diag(A_matrix)
    eigenvectors = H_matrices[0]
    for H in H_matrices[1:]:
        eigenvectors = multiply_matrices(eigenvectors, H)

    return eigenvalues, eigenvectors

# Пример использования
mtx = np.array([
    [-7, -5, -9],
    [-5, 5, 2],
    [-9, 2, 9]
], dtype=np.float64)

eigenvalues, eigenvectors = get_eigens(mtx)
print("Собственные значения:")
print(eigenvalues)
print("Собственные векторы:\n", eigenvectors, "\n")

numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(mtx)
print("Собственные значения (NumPy):", numpy_eigenvalues)
print("Собственные векторы (NumPy):\n", numpy_eigenvectors)
