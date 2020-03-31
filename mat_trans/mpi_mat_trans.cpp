#include<iostream>
#include <mpi.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>

// tags for different tasks
#define ROW_LEN_TAG 100
#define COLUMN_LEN_TAG 200
#define MATRIX_SENT_TAG 300
#define MATRIX_TRANS_SENT_TAG 400

using namespace std;

// print 2d matrix into matrix format
void print_mat_2d(int rows, int cols, int **data, string msg){
  cout << msg << endl;
  for (int irow = 0; irow < rows; irow++)
  {
    for (int icol = 0; icol < cols; icol++)
      cout << data[irow][icol] << " ";
    cout << "\n";
  }
  cout << endl;
}

// print 1d matrix into matrix format
void print_mat_1d(int rows, int cols, int *data, string msg){
  cout << msg << endl;
  for (int irow = 0; irow < rows; irow++)
  {
    for (int icol = 0; icol < cols; icol++)
      cout << data[irow*rows + cols] << " ";
    cout << endl;
  }
  cout << endl;
}

int main(int argc, char *argv[])
{
  int rank, size, temp;
  unsigned int ROW_LEN, COLUMN_LEN; //to store the column and row numbers of the matrix
  double starttime, endtime;
  int *dims, *periods, *coords;

  MPI_Comm MPI_Comm_cart;

  MPI_Comm MPI_Com_cart; //to store the cartesian communicator

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);


  int nodes_per_coord_axis = sqrt(25);

  dims = new int[2];
  periods = new int[2];
  coords = new int[2];

  dims[0] = dims[1] = nodes_per_coord_axis; //divide the whole matrix into equal no. of nodes
  periods[0] = periods[1] = 0;
  starttime = MPI_Wtime();

  //create new communicator with reordering of ranks enabled
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &MPI_Comm_cart);

  //get the new rank after cartesian topology implemented
  MPI_Comm_rank(MPI_Comm_cart, &rank);

  //get the cartesian coordinates of the specific rank
  MPI_Cart_coords(MPI_Comm_cart, rank, 2, coords);


  ROW_LEN = 625;
  COLUMN_LEN = 625;
  //for master rank
  if (coords[0] == 0 && coords[1] == 0)
  {
    // cout << "Master process" << endl;
    MPI_Status status;
    unsigned int sub_matrix_x = ROW_LEN / nodes_per_coord_axis; //colums in each sub-matrix
    unsigned int sub_matrix_y = COLUMN_LEN / nodes_per_coord_axis; //rows in each sub-matrix
    int **matrix;                         //to store the complete matrix read from the file
    int **matrixt;                        //to store the transposed matrix
    int *tempmatrix;                      //stores the matrix temporarily for transfer of submatrices
    tempmatrix = new int[sub_matrix_x * sub_matrix_y * sizeof(int)];

    //Original matrix memory allocation
    matrix = new int*[COLUMN_LEN * sizeof(int *)];
    for (int iter = 0; iter < COLUMN_LEN; iter++)
    {
      matrix[iter] = new int[ROW_LEN * sizeof(int)];
    }
    //Transposed matrix memory allocation
    matrixt = new int*[COLUMN_LEN * sizeof(int *)];
    for (int iter = 0; iter < ROW_LEN; iter++)
    {
      matrixt[iter] = new int[ROW_LEN * sizeof(int)];
    }


    //read matrix file 
    FILE *fileread = fopen("test_matrix", "r"); 


    for (int iter = 0; iter < COLUMN_LEN; iter++)
    {
      for (int iter2 = 0; iter2 < ROW_LEN; iter2++)
      {
        fscanf(fileread, "%d", &(matrix[iter][iter2]));
      }
    }
    // close file
    fclose(fileread);

    // cout << "File read successfully in master rank..." << endl;
    // print_mat_2d(ROW_LEN, COLUMN_LEN, matrix, "Original matrix");

    // to store the sub_matrix to be sent and transposed
    tempmatrix = new int[sub_matrix_x * sub_matrix_y * sizeof(int)];

    //send sub_matrix dimension info to other rank synchronously
    for (int cart_row = 0; cart_row < dims[1]; cart_row++)
    {
      for (int cart_column = 0; cart_column < dims[0]; cart_column++)
      {
        // this will be taken by master rank itself
        if (cart_row == 0 && cart_column == 0)
          continue;

        //figure out the rank required
        coords[0] = cart_column;
        coords[1] = cart_row;
        MPI_Cart_rank(MPI_Comm_cart, coords, &rank);

        //send the X and Y lengths of the matrix being sent
        MPI_Send(&sub_matrix_x, 1, MPI_UNSIGNED, rank, ROW_LEN_TAG, MPI_Comm_cart);
        MPI_Send(&sub_matrix_y, 1, MPI_UNSIGNED, rank, COLUMN_LEN_TAG, MPI_Comm_cart);

        //insert values to 1D array to be sent to the rank [iter2][iter]
        //values to be sent should be of [iter][iter2] part of the array since in the end we want a transpose
        temp = 0;
        for (int sub_mat_row = cart_row * sub_matrix_y; sub_mat_row < (cart_row * sub_matrix_y) + sub_matrix_y; sub_mat_row++)
        {
          for (int sub_mat_col = cart_column * sub_matrix_x; sub_mat_col < (cart_column * sub_matrix_x) + sub_matrix_x; sub_mat_col++)
          {
            tempmatrix[temp] = matrix[sub_mat_row][sub_mat_col];
            temp++;
          }
        }

        //send the sub-matrix to the specific rank
        int sum_matrix_len = sub_matrix_x * sub_matrix_y;
        // printf("\nSending to %d, %d\n------------\n", coords[0], coords[1]);
        MPI_Send(tempmatrix, sum_matrix_len, MPI_INT, rank, MATRIX_SENT_TAG, MPI_Comm_cart);
      }
    }
    //Transpose your own matrix
    for (int row = 0; row < sub_matrix_y; row++)
    {
      for (int col = 0; col < sub_matrix_x; col++)
      {
        matrixt[col][row] = matrix[row][col];
      }
    }

    //receive tranposed submatrix from slave ranks
    for (int cart_row = 0; cart_row < dims[1]; cart_row++)
    {
      for (int cart_col = 0; cart_col < dims[0]; cart_col++)
      {
        //this will be processed by the master rank itself
        if (cart_row == 0 && cart_col == 0)
          continue;

        coords[0] = cart_row;
        coords[1] = cart_col;
        MPI_Cart_rank(MPI_Comm_cart, coords, &rank);
        MPI_Recv(tempmatrix, sub_matrix_x * sub_matrix_y, MPI_INT, rank, MATRIX_TRANS_SENT_TAG, MPI_Comm_cart, &status);

        // printf("Received from %d\n", rank);
        temp = 0;
        for (int sub_mat_row = cart_row * sub_matrix_x; sub_mat_row < (cart_row * sub_matrix_x) + sub_matrix_x; sub_mat_row++)
        {
          for (int sub_mat_col = cart_col * sub_matrix_y; sub_mat_col < (cart_col * sub_matrix_y) + sub_matrix_y; sub_mat_col++)
          {
            matrixt[sub_mat_row][sub_mat_col] = tempmatrix[temp];
            //printf(" %d\t",matrixt[sub_mat_row][sub_mat_col]);
            temp++;
          }
          //printf("\n");
        }
      }
    }

    // cout << "Transformed matrix: " << endl;
    // for (int i = 0; i < ROW_LEN; i++)
    // {
    //   for (int j = 0; j < COLUMN_LEN; j++)
    //   {
    //     cout << matrixt[i][j] << " ";
    //   }
    //   cout << endl;
    // }

    // write the tranpose matrix into a file
    FILE *filewrite = fopen("test_matrix_output", "w");

    for (int iter = 0; iter < ROW_LEN; iter++)
    {
      for (int iter2 = 0; iter2 < COLUMN_LEN; iter2++)
      {
        fprintf(filewrite, "%d ", matrixt[iter][iter2]);
      }
      fprintf(filewrite, "\n");
    }
    // close file
    fclose(filewrite);

    endtime = MPI_Wtime();

    	printf("\nThe program took %f seconds\n",endtime-starttime);

    //de-allocating all memory
    for (int iter = 0; iter < COLUMN_LEN; iter++)
    {
      free(matrix[iter]);
    }
    free(matrix);

    for (int iter = 0; iter < ROW_LEN; iter++)
    {
      free(matrixt[iter]);
    }
    free(matrixt);

    free(tempmatrix);
  }
  else
  {
    // cout << "Slave process" << endl;
    // cout << coords[0] << " : " << coords[1] << endl;
    int x, y, temp;
    unsigned int sub_matrix_x; //colums in each sub-matrix
    unsigned int sub_matrix_y; //rows in each sub-matrix
    int *matrix1;                  //to store the complete submatrix sent by rank 0,0
    int *matrixt1;                 //to store the transposed submatrix
    
    MPI_Status status;
    int *coords2 = new int[2];
    //figure out the rank of coords 0,0
    coords[0] = 0;
    coords[1] = 0;

    MPI_Cart_coords(MPI_Comm_cart, rank, 2, coords2);
    MPI_Cart_rank(MPI_Comm_cart, coords, &rank);

    //recieve X and Y values
    MPI_Recv(&sub_matrix_x, 1, MPI_UNSIGNED, rank, ROW_LEN_TAG, MPI_Comm_cart, &status);
    MPI_Recv(&sub_matrix_y, 1, MPI_UNSIGNED, rank, COLUMN_LEN_TAG, MPI_Comm_cart, &status);

    //allocate 1D array accordingly matrix
    matrix1 = new int[sub_matrix_x * sub_matrix_y * sizeof(int)];

    //allocate 1D transformed array matrixt
    matrixt1 = new int[sub_matrix_x * sub_matrix_y * sizeof(int)];

    //recieve Matrix matrix
    temp = sub_matrix_x * sub_matrix_y;
    MPI_Recv(matrix1, temp, MPI_INT, rank, MATRIX_SENT_TAG, MPI_Comm_cart, &status);

    //tranform matrix to matrixt
    temp = 0;
    // cout << "received matrix from rank: " << rank << endl;

    for (x = 0; x < sub_matrix_x; x++)
    {
      for (y = x; y < sub_matrix_x * sub_matrix_y; y = y + sub_matrix_x){
        matrixt1[temp++] = matrix1[y];
        // cout << matrix1[y] << " ";
      }
      // cout << endl;
    }

    //send back matrix using tag MATRIX_TRANS_SENT_TAG
    MPI_Send(matrixt1, sub_matrix_x * sub_matrix_y, MPI_INT, rank, MATRIX_TRANS_SENT_TAG, MPI_Comm_cart);

    //de-allocate memory
    free(matrixt1);
    free(matrix1);
  }

  MPI_Finalize();
  return 0;
}