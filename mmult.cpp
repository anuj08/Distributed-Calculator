#include<fstream>
#include<string>


template <class T>
class Matrix {
    private:
        T *data;
        int row, int col;
    public:
        //filename, rows, columns
        Matrix(std::string filename, int r, int c){
            row = r;
            col = c;
            data = new T[rows*cols];
            std::ifstream fin;
            fin.open(filename);

            std::string line;
            int nums=0;
            if(fin.is_open()){
                while(getline(fin, line) && nums <= rows*cols){
                    
                }
            }
            
        }
        //filename, assume first line contains rows columns
        Matrix(std::string filename){

        }
};