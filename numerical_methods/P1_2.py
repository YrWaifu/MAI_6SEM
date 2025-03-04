# Реализовать метод прогонки в виде программы, задавая в качестве входных данных ненулевые элементы матрицы системы и вектор правых частей.
# Используя разработанное программное обеспечение, решить СЛАУ с трехдиагональной матрицей.

def thomas_algorithm(a, b, c, d):
    n = len(d)
    # Modify the coefficients
    c_prime = [0] * n
    d_prime = [0] * n

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / temp if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp

    # Back substitution
    x = [0] * n
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x

# Define the coefficients of the tridiagonal matrix
a = [0, -2, 2, -8, -7]  # Sub-diagonal (a1 is unused)
b = [8, 12, -9, 17, 13]  # Main diagonal
c = [-4, -7, 1, -4, 0]   # Super-diagonal (c5 is unused)
d = [32, 15, -10, 133, -76]  # Right-hand side

# Solve the system
solution = thomas_algorithm(a, b, c, d)
print("Solution:", solution)