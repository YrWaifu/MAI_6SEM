import numpy as np

def householder_qr(A):
    """
    QR-разложение с использованием преобразований Хаусхолдера.
    """
    m, n = A.shape
    R = A.astype(np.complex128)  # Копия матрицы A для преобразования
    Q = np.eye(m, dtype=np.complex128)  # Единичная матрица для накопления отражений

    for k in range(min(m - 1, n)):  # Проходим по столбцам
        # Выделяем подвектор из k-го столбца ниже диагонали
        x = R[k:, k]

        # Создаем вектор v для отражения Хаусхолдера
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1

        # Нормируем v
        v /= np.linalg.norm(v)

        # Применяем отражение к соответствующей части матрицы R
        R[k:, k:] -= 2 * np.outer(v, np.dot(v.conj(), R[k:, k:])) # H=I-2*(vvT/vTv)

        # Обновляем матрицу Q
        Q[:, k:] -= 2 * np.outer(Q[:, k:].dot(v), v.conj())

    return Q[:n], R[:n]


def qr_algorithm(A, tol=1e-15, max_iterations=1000):
    """
    QR-алгоритм для нахождения собственных значений матрицы A.
    """
    n = A.shape[0]
    Ak = A.astype(np.complex128)  # Работаем с комплексными числами
    counter = 0

    # Для отслеживания изменений собственных значений блоков 2x2
    prev_eigenvalues = None

    for iteration in range(max_iterations):
        counter += 1
        # Выполняем QR-разложение с помощью преобразований Хаусхолдера
        Q, R = householder_qr(Ak)

        # Обновляем матрицу Ak
        Ak = R @ Q

        # Проверяем условие сходимости норму поддиагональных элементов матрицы
        off_diagonal_norm = np.sqrt(np.sum(np.abs(np.tril(Ak, -1))**2))

        # Извлекаем текущие собственные значения
        current_eigenvalues = []
        i = 0
        while i < n:
            if i == n - 1 or abs(Ak[i + 1, i]) < tol:  # Одиночное вещественное значение
                current_eigenvalues.append(Ak[i, i])
                i += 1
            else:  # Блок 2x2 для комплексных значений
                submatrix = Ak[i:i+2, i:i+2]
                char_poly = np.poly(submatrix)
                roots = np.roots(char_poly)
                current_eigenvalues.extend(roots)
                i += 2

        # Преобразуем список в массив NumPy для удобства сравнения
        current_eigenvalues = np.array(current_eigenvalues)

        # Проверяем изменение собственных значений блоков 2x2
        if prev_eigenvalues is not None:
            # Сравниваем текущие и предыдущие собственные значения
            eigenvalue_diff = np.max(np.abs(current_eigenvalues - prev_eigenvalues))
            if eigenvalue_diff < tol:
                break

        # Обновляем предыдущие собственные значения
        prev_eigenvalues = current_eigenvalues.copy()

        # Проверяем классическое условие сходимости
        if off_diagonal_norm < tol:
            break

    print("Количество выполненных итераций:", counter)


    return current_eigenvalues


def format_complex(z, precision=2):
    """
    Форматирует комплексное число в виде 'a + bi' или 'a - bi' с округлением.
    """
    a = round(z.real, precision)
    b = round(z.imag, precision)

    if b == 0:
        return f"{a}"
    elif a == 0:
        return f"{b}i" if b > 0 else f"-{-b}i"
    else:
        return f"{a} {'+' if b > 0 else '-'} {abs(b)}i"


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
print("Собственные значения (NumPy):")

# Проверка корректности
print("Совпадают ли собственные значения?", np.allclose(np.sort(eigenvalues_qr), np.sort(eigenvalues_numpy)))