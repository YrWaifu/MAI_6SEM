import numpy as np


# Реализовать метод простых итераций и метод Зейделя в виде программ,
# задавая в качестве входных данных матрицу системы, вектор правых частей и точность вычислений.
# Используя разработанное программное обеспечение, решить СЛАУ.
# Проанализировать количество итераций, необходимое для достижения заданной точности.

def jacobi_method(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iterations

def gauss_seidel_method(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1
    return x, max_iterations

# Define the system of equations
A = np.array([
    [12, -3, -1, 3],
    [5, 20, 9, 1],
    [6, -3, -21, -7],
    [8, -7, 3, -27]
])
b = np.array([-31, 90, 119, 71])
x0 = np.zeros(len(b))
tol = 1e-5
max_iterations = 1000

# Solve using Jacobi Method
jacobi_solution, jacobi_iterations = jacobi_method(A, b, x0, tol, max_iterations)
print("Jacobi Method Solution:", jacobi_solution)
print("Jacobi Method Iterations:", jacobi_iterations)

# Solve using Gauss-Seidel Method
gs_solution, gs_iterations = gauss_seidel_method(A, b, x0, tol, max_iterations)
print("Gauss-Seidel Method Solution:", gs_solution)
print("Gauss-Seidel Method Iterations:", gs_iterations)