#include<fstream>
#include<string>
#include<sstream>
#include<mpi.h>
#include<iostream>

using namespace std;

template <class T>
class Matrix {
    private:
        T *data;
        int rows, cols;

    public:
        static int argc;
        static char **argv;
        static void init(int ac, char **av){
            argc = ac;
            argv = av;
        }
        //filename, rows, columns, stored in row major format
        Matrix(std::string filename, int r, int c){
            rows = r;
            cols = c;
            data = new T[rows*cols];
            std::ifstream fin;
            fin.open(filename);

            std::string line;
            int i=0;
            if(fin.is_open()){
                while(getline(fin, line) && i <= rows*cols){
                    std::istringstream iss(line);
                    while(i < rows*cols && iss >> data[i]){
                        i++;
                    }
                }
            }
            fin.close();
        }
        //filename, assume first line contains rows columns
        Matrix(std::string filename){
            std::ifstream fin;
            fin.open(filename);
            std::string line;
            if(fin.is_open()){
                getline(fin, line);
                std::istringstream iss(line);
                iss >> rows;
                iss >> cols;
                data = new T[rows*cols];
                int i=0;
                while(getline(fin, line) && i <= rows*cols){
                    std::istringstream iss(line);
                    while(i < rows*cols && iss >> data[i]){
                        i++;
                    }
                }
            }
            fin.close();
        }

        T& at(int r, int c){
            return data[r*rows + c];
        }

        void print(){
            for(int i=0; i<rows; ++i){
                for(int j=0; j<cols; ++j){
                    cout<<data[i*rows + j]<<" ";
                }
                std::cout<<std::endl;
            }
        }

        //Take transpose of a matrix
        void transpose(){
          T *new_array = new T[rows*cols];
          for (int i = 0; i < rows; ++i)
          {
            for (int j = 0; j < cols; ++j)
            {
              // Index in the original matrix.
              int index1 = i * cols + j;

              // Index in the transpose matrix.
              int index2 = j * rows + i;

              new_array[index2] = data[index1];
            }
          }

          for (int i = 0; i < rows * cols; i++)
          {
            data[i] = new_array[i];
          }
        }

        MPI_Datatype get_type(){
            char name = typeid(T).name()[0];
            switch (name) {
                case 'i':
                    return MPI_INT;
                    break;
                case 'f':
                    return MPI_FLOAT;
                    break;
                case 'j':
                    return MPI_UNSIGNED;
                    break;
                case 'd':
                    return MPI_DOUBLE;
                    break;
                case 'c':
                    return MPI_CHAR;
                    break;
                case 's':
                    return MPI_SHORT;
                    break;
                case 'l':
                    return MPI_LONG;
                    break;
                case 'm':
                    return MPI_UNSIGNED_LONG;
                    break;
                case 'b':
                    return MPI_BYTE;
                    break;
            }
        }
};
