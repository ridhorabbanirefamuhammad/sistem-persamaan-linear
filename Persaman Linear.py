import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def plot_solution_line(x, method_name):
    labels = [f'x{i+1}' for i in range(len(x))]
    plt.plot(labels, x, marker='o', linestyle='-', label=method_name)
    plt.xlabel('Variabel')
    plt.ylabel('Nilai Solusi')
    plt.title('Visualisasi Solusi dalam Plot Garis')
    plt.grid(True)
    plt.legend()
    plt.show()

def solve_using_inverse_matrix(A, b):
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x, "Matriks Balikan"

def solve_using_lu_gauss(A, b):
    LU, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((LU, piv), b)
    return x, "Dekomposisi LU Gauss"

def solve_using_crout(A, b):
    P, L, U = scipy.linalg.lu(A)
    x = scipy.linalg.solve_triangular(L, np.dot(P, b), lower=True)
    x = scipy.linalg.solve_triangular(U, x)
    return x, "Dekomposisi Crout"

n = int(input("Masukkan jumlah variabel: "))  
A = np.zeros((n, n))

print("Masukkan matriks koefisien A:")
for i in range(n):
    A[i] = [float(x) for x in input().split()]

print("Masukkan vektor hasil b:")
b = [float(x) for x in input().split()]

x_inverse, method_name_inverse = solve_using_inverse_matrix(A, b)
x_lu, method_name_lu = solve_using_lu_gauss(A, b)
x_crout, method_name_crout = solve_using_crout(A, b)

print("Solusi x menggunakan matriks balikan:", x_inverse)
print("Solusi x menggunakan dekomposisi LU Gauss:", x_lu)
print("Solusi x menggunakan dekomposisi Crout:", x_crout)

plot_solution_line(x_inverse, method_name_inverse)
plot_solution_line(x_lu, method_name_lu)
plot_solution_line(x_crout, method_name_crout)
