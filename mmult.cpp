#include<fstream>
#include<string>
#include<sstream>
#include<iostream>

using namespace std;

template <class T>
class Matrix {
    private:
        T *data;
        int rows, cols;

    public:
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
};