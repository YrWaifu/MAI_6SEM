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
    for i in range(n - 1):  # Проходим по строкам
        for j in range(i + 1, n):  # Проходим по столбцам выше главной диагонали
            if abs(mtx[i, j]) > max_val:  # Ищем максимальный по модулю элемент
                max_val = abs(mtx[i, j])
                indices = (i, j)
    return indices

# Функция для получения матрицы вращения H
def get_h_matrix(mtx, indices):
    # Строит матрицу вращения H, которая обнуляет максимальный недиагональный элемент.
    i_ind, J_ind = indices
    # Вычисляем угол поворота
    rotation_angle = 0.5 * np.arctan2(2 * mtx[i_ind, J_ind], mtx[i_ind, i_ind] - mtx[J_ind, J_ind])
    # Создаем единичную матрицу
    H = np.eye(mtx.shape[0])
    # Заполняем элементы матрицы вращения
    H[i_ind, i_ind] = H[J_ind, J_ind] = np.cos(rotation_angle)
    H[i_ind, J_ind] = -np.sin(rotation_angle)
    H[J_ind, i_ind] = np.sin(rotation_angle)
    return H

# Функция для вычисления ошибки
def calculate_the_error(mtx):
    # Вычисляет ошибку как норму всех недиагональных элементов матрицы.
    error = 0
    n = mtx.shape[0]
    for i in range(n - 1):  # Проходим по строкам
        for j in range(i + 1, n):  # Проходим по столбцам выше главной диагонали
            error += mtx[i, j]**2  # Суммируем квадраты недиагональных элементов
    return np.sqrt(error)  # Возвращаем корень из суммы

# Функция для нахождения собственных значений и векторов
def get_eigens(mtx, epsilon=1e-9):
    # Находит собственные значения и собственные векторы симметричной матрицы методом вращений.
    A_matrix = mtx.copy()  # Копируем исходную матрицу
    H_matrices = []  # Список для хранения матриц вращения
    stop = False  # Флаг завершения цикла
    while not stop:
        indices = get_indices(A_matrix)  # Находим индексы максимального недиагонального элемента
        H = get_h_matrix(A_matrix, indices)  # Строим матрицу вращения
        H_matrices.append(H)  # Добавляем матрицу вращения в список
        new_A_matrix = multiply_matrices(transpose_matrix(H), A_matrix)  # Применяем преобразование
        A_matrix = multiply_matrices(new_A_matrix, H)  # Обновляем матрицу A
        if calculate_the_error(A_matrix) < epsilon:  # Проверяем условие завершения (сумма квадратов вндиагональных элементов)
            stop = True

    # Собственные значения находятся на диагонали матрицы A
    eigenvalues = np.diag(A_matrix)
    # Собственные векторы получаются перемножением всех матриц вращения
    eigenvectors = H_matrices[0]
    for H in H_matrices[1:]:
        eigenvectors = multiply_matrices(eigenvectors, H)

    return eigenvalues, eigenvectors

# Пример использования
mtx = np.array([
    [4, 7, -1],
    [7, -9, -6],
    [-1, -6, -4]
], dtype=np.float64)

# Находим собственные значения и векторы
eigenvalues, eigenvectors = get_eigens(mtx)
print("Собственные значения:")
print(eigenvalues)
print("Собственные векторы:\n", eigenvectors, "\n")

# Сравниваем результаты с NumPy
numpy_eigenvalues, numpy_eigenvectors = np.linalg.eigh(mtx)
print("Собственные значения (NumPy):", numpy_eigenvalues)
print("Собственные векторы (NumPy):\n", numpy_eigenvectors)