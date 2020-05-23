
### Use user_la.py file to test the Distributed Algorithms
### update to hosts file configuration which are on the same network in the following format: 
- [ip address] [username]

## Methods implemented for the end user 
- Distr_LA.configure(hostfile, number of processes): To update the host file name and total number of processess spawn for the computation. 
- Distr_LA.distr_gauss(a, b, [optional parameter: number of processes]): To compute the solution to Ax = b using gaussian elimination. 
- Distr_LA.distr_mult(a, b, [optional parameter: number of processes]): To compute the matrix multiplication of A and B
- Distr_LA.distr_det(a, [optional parameter: number of processes]): To compute the determinant of the matrix A.
- Distr_LA.distr_inverse(a, [optional parameter: number of processes]): To compute the inverse of a matrix A.
- Distr_LA.distr_transpose(a, [optional parameter: number of processes]): Distributed transpose of a Matrix A
- Distr_LA.distr_qr(a, [optional parameter: number of processes]): Distributed QR decomposition of a Matrix A
- Distr_LA.distr_ei(a, [optional parameter: number of processes]): Distributed eigen values and eigen vectors of a Matrix A



## To test the correction of the distributed algorithms 
Run: 
```
python3 test_la.py
```

## To run the program for  distributed algorithms 
Run: 
```
python3 user_la.py
```