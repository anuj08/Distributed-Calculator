#include "mmult.cpp"

using namespace std;
template<class T>
Matrix<T> Matrix<T>::multiply(Matrix<T> &B){
    int size, rank;
    // MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int sobuf, sobufb; 
    int disCounts[size];
    int offsets[size];
    int base, extra;
    int initOff = 0;
    int tmp, sb;
    int r,c;
    T *Bdata;


    if(rank == 0){
        //Row splitting
        c = this->getNumColumns();
        r = this->getNumRows();
        Bdata = B.data;
        sb = B.getNumColumns()*B.getNumRows();
    }

    MPI_Barrier(MPI_COMM_WORLD);        
    
    MPI_Bcast(&r, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_INT,  0, MPI_COMM_WORLD);

    base = r/size;
    extra = r%size;

    for(int i=0; i<size; ++i){
        tmp = base + (extra-- > 0);
        disCounts[i] = tmp*c;
        offsets[i] = initOff;
        initOff += tmp*c;
    }
    MPI_Bcast(&sb, 1, MPI_INT,  0, MPI_COMM_WORLD);

    sobuf = (base+1)*c;
    sobufb = (base+1)*(sb/c);

    if(rank != 0){
        Bdata = new T[sb];
    }

    T aa[sobuf];
    T cc[sobufb];
    T *C;

    MPI_Scatterv(this->data, disCounts, offsets, this->get_type(), aa, sobuf, this->get_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(Bdata, B.getNumColumns()*B.getNumRows(), B.get_type(), 0, MPI_COMM_WORLD);

    int sum = 0;
    int loops = disCounts[rank];

    for(int k=0; k<disCounts[rank]/c; ++k){
        for(int i=0; i<(sb/c); ++i){
            for(int j=0; j<c; ++j){
                sum += aa[j + k*c]*Bdata[j*(sb/c) + i];
            }
            cc[i + k*(sb/c)] = sum;
            sum = 0;    
        }
    }

    if(rank == 0){
        C = new T[this->getNumRows()*B.getNumColumns()];
    }

    base = r/size;
    extra = r%size;
    initOff = 0;
    for(int i=0; i<size; ++i){
        tmp = base + (extra-- > 0);
        disCounts[i] = tmp*(sb/c);
        offsets[i] = initOff;
        initOff += tmp*(sb/c);
    }

    MPI_Gatherv(cc, disCounts[rank], this->get_type(), C, disCounts, offsets, this->get_type(), 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);        

    if(rank == 0){
        Matrix<int> Cmat(C,r,sb/c);
        return Cmat;
    }
}
