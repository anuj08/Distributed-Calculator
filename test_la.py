from custom_la import Distr_LA
import numpy as np
import random

#  create configuration
# Distr_LA.configure("hosts", 2)

random.seed(0)
N = 10
a = np.zeros((N, N), dtype='float64')
b = np.zeros((N, N), dtype='float64')
cb = np.zeros((N, 1), dtype='float64')

for i in range(N):
    cb[i, 0] = random.randint(10, 100)
    for j in range(N):
        a[i,j] = random.randint(10,100)
        b[i,j] = random.randint(10,100)


# for gaussian testing: 
distr_val = Distr_LA.distr_gauss(a, cb, 3)
local_val = np.linalg.solve(a, cb)
print("Gaussian eliminiation results comparison: ", np.allclose(distr_val, local_val))

#  matrix multiplication test
distr_val = Distr_LA.distr_mult(a, b, 3)
local_val = np.dot(a, b)
print("Matrix Multiplication results comparison: ", np.allclose(distr_val, local_val))


# matrix determinant  test
distr_val = Distr_LA.distr_det(a, 3)
# print(distr_val)
local_val = np.linalg.det(a)
# print(local_val)
print("Determinant results comparison: ", np.allclose(distr_val, local_val))

#  matrix inverse test
distr_val = Distr_LA.distr_inverse(a, 3)
local_val = np.linalg.inv(a)
print("Matrix inverse results comparison: ", np.allclose(distr_val, local_val))


# transpose testing
distr_val = Distr_LA.distr_transpose(a, 3)
# print(distr_val)
# print(a)
local_val = a.transpose()
print("Matrix transpose results comparison: ", np.allclose(distr_val, local_val))


# qr testing
distr_val_Q, distr_val_R = Distr_LA.distr_qr(a, 3)

print("QR decomposition results comparison: ", np.allclose(a, np.dot(distr_val_Q, distr_val_R)))


#  eigen vector and eigen value test
a = np.array([[1, 3, 0], [3, 2, 1], [0, 1, 3]])
distr_val, distr_val_vec = Distr_LA.distr_ei(a, 3)

satisfied = True
for i in range(len(distr_val)):
  # print("For eigen values: ", distr_val[i])
  v1 = np.dot(a, distr_val_vec[:, i])
  v2 = distr_val[i] * distr_val_vec[:, i]
  # print(v1)
  # print(v2)
  if np.allclose(v1, v2, rtol=.01):
    # print("True")
    continue
  satisfied = False


print("Eigen values and Eigen vector test results: ", satisfied)


