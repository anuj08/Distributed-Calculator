#include "mmult.cpp"

using namespace std;
template<class T>
Matrix<T> Matrix<T>::multiply(Matrix<T> &B, int argc, char **argv){
    int size, rank;
    MPI_Init(&argc, &argv);
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
    MPI_Bcast(&r, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_INT,  0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    base = r/size;
    extra = r%size;

    for(int i=0; i<size; ++i){
        tmp = base + (extra-- > 0);
        disCounts[i] = tmp*c;
        offsets[i] = initOff;
        initOff += tmp*c;
        // cout<<i<<" "<<disCounts[i]<<" "<<offsets[i]<<endl;
    }



        // cout<<"data is"<<endl;
        // for(int i=0; i<r*c; ++i){
        //     cout<<this->data[i]<<" ";
        // }
        // cout<<endl;


    MPI_Bcast(&sb, 1, MPI_INT,  0, MPI_COMM_WORLD);


    sobuf = (base+1)*c;
    sobufb = (base+1)*(sb/c);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0){
        Bdata = new T[sb];
    }
    // MPI_Bcast(&sobuf, 1, MPI_INT,  0, MPI_COMM_WORLD);

    T aa[sobuf];
    T cc[sobufb];
    T *C;

    MPI_Scatterv(this->data, disCounts, offsets, this->get_type(), aa, sobuf, this->get_type(), 0, MPI_COMM_WORLD);
    // if(rank != 0){
    //     for(int i=0; i<disCounts[rank]; ++i){
    //         // cout<<aa[i]<<" ";
    //     }
    //     cout<<endl;
    //     cout<<disCounts[rank]<<endl;

    // }

    MPI_Bcast(Bdata, B.getNumColumns()*B.getNumRows(), B.get_type(), 0, MPI_COMM_WORLD);

    int sum = 0;
    int loops = disCounts[rank];


    MPI_Barrier(MPI_COMM_WORLD);
    // cout<<rank<<" :"<<sobuf<<" :"<<disCounts[rank]<<endl;
    // if(rank != 0){
        // for(int i=0; i<sobuf; ++i){
        //     cout<<Bdata[i]<<" ";
        // }
        // cout<<endl;
        // cout<<disCounts[rank]<<endl;
    // }

    // cout<<"FIGS: ";
    // cout<<r<<" "<<c<<" "<<sb/c<<endl;

    for(int k=0; k<disCounts[rank]/c; ++k){
        for(int i=0; i<(sb/c); ++i){
            for(int j=0; j<c; ++j){
                sum += aa[j + k*c]*Bdata[j*(sb/c) + i];
                // cout<<aa[j]<<" "<<Bdata[j*c + i]<<endl;
            }
            cc[i + k*(sb/c)] = sum;
            sum = 0;    
            // cout<<cc[i + k*(sb/c)]<<" ";
        }
    }
    // cout<<"END OF COMP FOR: "<<rank<<endl;

    MPI_Barrier(MPI_COMM_WORLD);
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
        // cout<<i<<" "<<disCounts[i]<<" "<<offsets[i]<<endl;
    }
    // cout<<"Before mpi gather"<<endl;
    MPI_Gatherv(cc, disCounts[rank], this->get_type(), C, disCounts, offsets, this->get_type(), 0, MPI_COMM_WORLD);

    // cout<<"After mpi gather"<<endl;

    MPI_Barrier(MPI_COMM_WORLD);        
    MPI_Finalize();

    if(rank == 0){
        for(int i=0; i<this->getNumRows(); ++i){
            for(int j=0; j<B.getNumColumns(); ++j){
                cout<<C[i*(sb/c) + j]<<" ";
            }
            cout<<endl;
        }
    }
}
