#include <ctime>
#include <stdio.h>
#include <time.h>
#include <chrono>

#define CUDA_CALL(f)                                                                     \
    {                                                                                    \
        cudaError_t err = f;                                                             \
        if(err != cudaSuccess){                                                           \
            fprintf(stderr, "%s\n at %s:%d\n", cudaGetErrorName(err), __FILE__, __LINE__);\
            exit(1);\
        }\
    }\

struct Vec{
    float * data;
    size_t n;
};

float get_rand(){
    return float(rand()) / RAND_MAX;
};

void vec_add_cpu(Vec * a, Vec * b, Vec * c){
    for(int i = 0; i < a->n; i++){
        c->data[i] = a->data[i] + b->data[i];
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

#define N (1000000 * 20)

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

bool verify(Vec * cpu_res, float * gpu_res){
    float eps = 1e-5;
    for(int i = 0; i < cpu_res->n; i++){
        if(fabs(cpu_res->data[i] - gpu_res[i]) > eps){
            return false;
        }
    }
    return true;
};

double benchmark_cpu(Vec * a, Vec * b, Vec * c, int num_runs){
    timespec start, end;
    double cpu_elapsed_avg = 0.0f;

    for(int i = 0; i < num_runs; i++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        vec_add_cpu(a, b, c);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = get_elapsed(start, end);
        cpu_elapsed_avg += elapsed;
    }
    cpu_elapsed_avg /= num_runs;

    return cpu_elapsed_avg;
};

double benchmark_gpu(float * d_a, float * d_b, float * d_c, int num_runs){
    size_t block_size = 256;
    size_t num_blocks = (N + block_size  -1) / block_size;
    timespec start, end;
    double avg_elapsed;

    for(int i = 0; i < num_runs; i++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        vec_add_gpu<<<num_blocks, block_size>>>(d_a, d_b, d_c, N);
        CUDA_CALL(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = get_elapsed(start, end);
        avg_elapsed += elapsed;
    }
    avg_elapsed /= num_runs;
    return avg_elapsed;
};

int main(){
    srand(time(0));
    Vec a, b, c;

    vec_init_rand(&a, N);
    vec_init_rand(&b, N);
    vec_init_zeros(&c, N);
    vec_add_cpu(&a, &b, &c);

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

    printf("Success: %d\n", verify(&c, res.data));
    double cpu_elapsed_avg = benchmark_cpu(&a, &b, &c, 10);
    double gpu_elapsed_avg = benchmark_gpu(d_a, d_b, d_c, 10);

    printf("elapsed time cpu: %lf\n", cpu_elapsed_avg);
    printf("elapsed time gpu: %lf\n", gpu_elapsed_avg);
    printf("gpu is %f times faster than cpu\n", cpu_elapsed_avg / gpu_elapsed_avg);


    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
    CUDA_CALL(cudaFree(d_c));

    vec_free(&c);
    vec_free(&a);
    vec_free(&b);
}
