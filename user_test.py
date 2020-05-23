from custom_la import Distr_LA
import numpy as np
import random

#  create configuration
Distr_LA.configure("hosts")

random.seed(0)
N = 1000
a = np.zeros((N, N), dtype='float64')
b = np.zeros((N, N), dtype='float64')

for i in range(N):
    for j in range(N):
        a[i,j] = random.randint(1,100)
        b[i,j] = random.randint(1,100)

print("Matrix multiplication : ", Distr_LA.distr_mult(a, b, 13))

#matrix determinant  test
# a = np.array([[0, 0, 1], [1, 1, 3], [1, 2, 4]], dtype='float64')
# print("solution is : ", Distr_LA.distr_det(a, 3))

#  matrix inverse test
# a = np.array([[0, 0, 1], [1, 1, 3], [1, 2, 4]], dtype='float64')
# print("solution is : ", Distr_LA.distr_inverse(a, 3))

# transpose testing
# a = np.array([[1, 3, 0], [3, 2, 1], [0, 1, 3]])
# print("Transposed matrix: ", Distr_LA.distr_transpose(a, 3))

# qr testing
# a = np.array([[1, 3, 0], [3, 2, 1], [0, 1, 3]])
# print("QR decomposition ", Distr_LA.distr_qr(a, 1))

#  eigen vector and eigen value test
# a = np.array([[1, 3, 0], [3, 2, 1], [0, 1, 3]])
# print("Eigen values and vectors: ", Distr_LA.distr_ei(a, 3))


