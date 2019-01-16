/////////////////////////////////////////////////////////////////
//matrix multiplication is performed in this lab               //
//                                                             //
//for run:                                                     //
//1) make                                                      //
//2) ./makefile                                                //
//                                                             //
/////////////////////////////////////////////////////////////////


#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 122;
    int SIZE = N*N;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = i+i*j;
            h_B[i*N+j] = i-i*j);
        }
    }
    // Print elements matrix
    for (int  i=0; i<N*N; i++){
    	cout<<h_A[i]<<endl;
    }
    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);
    clock_t start, end;
    start = clock();
    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();
    end = clock();
    cout<<"time for matrixMul in GPU: "<< end-start<<endl;
    float *cpu_C;
    cpu_C=new float[SIZE];
    start = clock();
    // Now do the matrix multiplication on the CPU
    float sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }
    end = clock();
    cout<<"time for matrixMul in CPU: "<< end-start<<endl;
    cout << "cpu_C: "<< cpu_C << endl;
    return 0;
}

