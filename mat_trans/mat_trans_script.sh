
## Compilation
g++ ./gen_test_matrix.cpp -o gen_obj
mpic++ mpi_mat_trans.cpp  -o mpi_obj

## save the input matrix
./gen_obj 625 > test_matrix

## exec mpi file to take transpose of a input matrix
mpiexec -np 25 --hostfile hostfile ./mpi_obj


## Delete files
# rm gen_obj test_matrix test_matrix_output mpi_obj