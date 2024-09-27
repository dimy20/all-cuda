#include <stdio.h>
#include "err.h"

struct Mat{
    float * data;
    int n;
    int m;
};

float get_rand(){
    return float(rand()) / RAND_MAX;
};

void mat_init_zeros(Mat * mat, int n, int m){
    mat->data = (float *)malloc(sizeof(float) * n * m);

    if(mat->data == NULL){
        fprintf(stderr, "Error: Failed to allocate matrix of (%d, %d)\n", n, m);
        exit(1);
    }

    memset(mat->data, 0, sizeof(float) * n * m);
    mat->n = n;
    mat->m = m;
}

void mat_init_with_value(Mat *mat, int n, int m, float value){
    mat_init_zeros(mat, n, m);
    for(int i = 0; i < mat->n; i++){
        for(int j = 0; j < mat->m; j++){
            mat->data[i * mat->m + j] = value;
        }
    }
}

void mat_init_rand(Mat * mat, int n , int m){
    mat_init_zeros(mat, n, m);
    for(int i = 0; i < mat->n; i++){
        for(int j = 0; j < mat->m; j++){
            float x = get_rand();
            mat->data[i * mat->m + j] = x;
        }
    }
}

void mat_init_with_values(Mat * mat, int n, int m, const float * values){
    mat_init_zeros(mat, n, m);
    memcpy(mat->data, values, sizeof(float) * n * m);
};

void mat_free(Mat * mat){
    if(mat->data != NULL){
        free(mat->data);
    }
}

void mat_print(Mat * mat){
    for(int i = 0; i < mat->n; i++){
        for(int j = 0; j < mat->m; j++){
            float x = mat->data[i * mat->m + j];
            printf(j < mat->m - 1 ? "%f, " : "%f", x);
        }
        printf("\n");
    }
    printf("\n");
};

void matmul_cpu(Mat * A, Mat * B, Mat * C){
    if(A->m != B->n){
        fprintf(stderr, "Invalid Matrix shapes");
        exit(1);
    }

    for(int i = 0; i < C->n; i++){
        for(int j = 0; j < C->m; j++){
            for(int k = 0; k < A->m; k++){
                C->data[i * C->m + j] += A->data[i * A->m + k] * B->data[k * B->m + j];
            }
        }
    }
}

__global__ void matmul_gpu(float * A, float * B, float * C, int n, int m, int p){
    int block_offset = blockIdx.y + gridDim.x + blockIdx.x;
    int num_threads_per_block = blockDim.x * blockDim.y;
    int thread_offset = block_offset * num_threads_per_block;

    int i = thread_offset + (threadIdx.y * blockDim.x);
    int j = thread_offset + threadIdx.x;
    if(i < n && j < m){
        C[i * m + j] = 0;
        for(int k = 0; k < p; k++){
            C[i * m + j] += A[i * p + k] * B[k * p + j];
        }
    }
};

bool verify(float * A, float * res, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(fabs(A[i * m + j]- res[i * m + j]) > 1e-5){
                return false;
            }
        }
    }
    return true;
};

int main(){
    Mat A, B, C;
    int n, m, p;
    n = 1024;
    p = 512;
    m = 256;

    mat_init_rand(&A, n, p);
    mat_init_rand(&B, p, m);
    mat_init_zeros(&C, n, m);

    //mat_print(&A);
    //mat_print(&B);
    matmul_cpu(&A, &B, &C);
    //mat_print(&C);
    
    //setup cuda stuff

    float * d_A, * d_B, * d_C;

    CUDA_CALL(cudaMalloc(&d_A, sizeof(float) * A.n * A.m));
    CUDA_CALL(cudaMalloc(&d_B, sizeof(float) * B.n * B.m));
    CUDA_CALL(cudaMalloc(&d_C, sizeof(float) * C.n * C.m));

    CUDA_CALL(cudaMemcpy(d_A, A.data, sizeof(float) * A.n * A.m, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, B.data, sizeof(float) * B.n * B.m, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_C, C.data, sizeof(float) * C.n * C.m, cudaMemcpyHostToDevice));

    int threads_per_dim = 256;
    dim3 block_size(threads_per_dim, threads_per_dim);
    int grid_X = (n + block_size.x - 1) / block_size.x;
    dim3 grid_size(grid_X, grid_X, 1);

    matmul_gpu<<<grid_size, block_size>>>(d_A, d_B, d_C, n, m, p);
    CUDA_CALL(cudaDeviceSynchronize());

    float res[n*m];
    CUDA_CALL(cudaMemcpy(res, d_C, sizeof(float) * n * m, cudaMemcpyDeviceToHost));

    //Mat R;
    //R.data = res;
    //R.m = m;
    //R.n = n;

    //mat_print(&R);

    printf("Success: %d\n" , verify(C.data, res, n, m));

    mat_free(&A);
    mat_free(&B);
    mat_free(&C);

    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
};
