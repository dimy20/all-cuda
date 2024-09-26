#include <ctime>
#include <stdio.h>
#include <time.h>
#include <chrono>

struct Vec{
    float * data;
    size_t n;
};

float get_rand(){
    return float(rand()) / RAND_MAX;
};

void vec_add_cpu(float * a, float * b, float * c, size_t n){
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
};

void vec_init(Vec * v, size_t n){
    v->data = (float *)malloc(sizeof(float) * n);
    v->n = n;
    if(!v->data){
        fprintf(stderr, "failed to allocate vector\n");
        exit(1);
    }
}

void vec_free(Vec * v){
    if(v->data != NULL){
        free(v->data);
    }
}

void vec_init_rand(Vec * v, size_t n){
    vec_init(v, n);
    for(int i = 0; i < v->n; i++){
        v->data[i] = get_rand();
    };
};

void vec_init_zeros(Vec * v, size_t n){
    vec_init(v, n);
    memset(v->data, 0, sizeof(float) * v->n);
}

void vec_print(const Vec * v){
    printf("[");
    for(int i = 0; i < v->n; i++){
        if(i <= v->n - 2){
            printf("%f, ", v->data[i]);
        }else{
            printf("%f", v->data[i]);
        }
    }
    printf("]\n");
};

#define N 1024

#define CUDA_CALL(f)                                                                     \
    {                                                                                    \
        cudaError_t err = f;                                                             \
        if(err != cudaSuccess){                                                           \
            fprintf(stderr, "%s\n at %s:%d\n", cudaGetErrorName(err), __FILE__, __LINE__);\
            exit(1);\
        }\
    }\

__global__ void vec_add_gpu(float * a, float * b, float * c, size_t n){
    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    int id = blockIdx.x * num_threads + threadIdx.x;

    if(id < n){
        c[id] = a[id] + b[id];
    };
};

double get_elapsed(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(){
    srand(time(0));
    Vec a, b, c;

    vec_init_rand(&a, N);
    vec_init_rand(&b, N);
    vec_init_zeros(&c, N);

    //vec_print(&a);
    //vec_print(&b);

    timespec start, end;
    double cpu_elapsed_avg = 0.0f;

    for(int i = 0; i < 20; i++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        vec_add_cpu(a.data, b.data, c.data, c.n);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = get_elapsed(start, end);
        cpu_elapsed_avg += elapsed;
    }

    printf("elapsed time cpu: %lf\n", cpu_elapsed_avg);
    
    //vec_print(&c);

    float * d_a, * d_b, * d_c;

    CUDA_CALL(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_c, N * sizeof(float)));

    CUDA_CALL(cudaMemcpy(d_a, a.data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_b, b.data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_c, c.data, N * sizeof(float), cudaMemcpyHostToDevice));

    size_t block_size = 256;
    size_t num_blocks = (N + block_size  -1) / block_size;

    vec_add_gpu<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);

    CUDA_CALL(cudaDeviceSynchronize());

    Vec res;
    vec_init_zeros(&res, N);
    CUDA_CALL(cudaMemcpy(res.data, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    //vec_print(&res);

    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
    CUDA_CALL(cudaFree(d_c));

    vec_free(&c);
    vec_free(&a);
    vec_free(&b);
}
