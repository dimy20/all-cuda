#include <stdio.h>

__global__ void whoami(){
    //skips z times xy

    int z_offset = gridDim.x * gridDim.y * blockIdx.z;
    int x_offset = gridDim.x * blockIdx.y;
    int block_id = z_offset + x_offset + blockIdx.x;

    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int block_offset =  threads_per_block * block_id;

    int tz_offset = blockDim.x * blockDim.y * threadIdx.z;
    int tx_offset = blockDim.x * threadIdx.y;

    int thread_id = tz_offset + tx_offset + block_offset + threadIdx.x;

    printf("(Block=%d, thread=%d)\n", block_id, thread_id);
};

int main(){
    int b_x = 2;
    int b_y = 3;
    int b_z = 4;
    int t_x = 4, t_y = 4, t_z = 4;

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_x * t_y;

    printf("blocks per grid : %d\n", blocks_per_grid);
    printf("threads per block: %d\n", blocks_per_grid);
    printf("total threads %d\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);


    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    return 0;
};
