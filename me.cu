#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "me.h"
#include "tables.h"


int *d_best_sad, *d_best_x, *d_best_y; 
uint8_t *d_orig_y, *d_orig_u, *d_orig_v;
uint8_t *d_ref_y, *d_ref_u, *d_ref_v;



void initialize_device_memory(struct c63_common *cm)
{
  int total_size_y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT];
  int total_size_u = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT];
  int total_size_v = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT];

  cudaMalloc(&d_orig_y, total_size_y * sizeof(uint8_t));
  cudaMalloc(&d_ref_y, total_size_y * sizeof(uint8_t));

  cudaMalloc(&d_orig_u, total_size_u * sizeof(uint8_t));
  cudaMalloc(&d_ref_u, total_size_u * sizeof(uint8_t));

  cudaMalloc(&d_orig_v, total_size_v * sizeof(uint8_t));
  cudaMalloc(&d_ref_v, total_size_v * sizeof(uint8_t));

  // Allocate memory for best_sad, best_x, best_y on device
  cudaMalloc(&d_best_sad, sizeof(int));
  cudaMalloc(&d_best_x, sizeof(int));
  cudaMalloc(&d_best_y, sizeof(int));
}


void cleanup_device_memory()
{
    // Free each allocated device pointer for original and reference frames
    cudaFree(d_orig_y);
    cudaFree(d_orig_u);
    cudaFree(d_orig_v);

    cudaFree(d_ref_y);
    cudaFree(d_ref_u);
    cudaFree(d_ref_v);

    // Free device pointers for SAD calculations
    cudaFree(d_best_sad);
    cudaFree(d_best_x);
    cudaFree(d_best_y);
}



__global__ void computeSADIntegrated(int left, int right, int top, int bottom, int w, int mx, int my, uint8_t *cu_offset, uint8_t *cu_ref, int *best_sad, int *best_x, int *best_y)
{
    int x = left + blockIdx.x * blockDim.x + threadIdx.x;
    int y = top + blockIdx.y * blockDim.y + threadIdx.y;

    if (x < right && y < bottom) {
        __shared__ int sad_shared[64];
        int i = threadIdx.y * blockDim.x + threadIdx.x;  // Thread index in the block

        if (i < 64) {
            int bx = i % 8;  // Block x index
            int by = i / 8;  // Block y index
            int index = (y + by) * w + (x + bx);
            sad_shared[i] = abs(cu_ref[index] - cu_offset[index]);
        }

        __syncthreads(); // Ensure all threads have written their SAD values

        // Reduction within the block
        int block_sad = 0;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            for (int j = 0; j < 64; ++j) {
                block_sad += sad_shared[j];
            }

            // Atomic update for best sad and corresponding coordinates
            if (atomicMin(best_sad, block_sad) == block_sad) {
                *best_x = x - mx;
                *best_y = y - my;
            }
        }
    }
}





/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
  int color_component, uint8_t *d_orig, uint8_t *d_ref)
{
    struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];
    int range = cm->me_search_range;
    if (color_component > 0) { range /= 2; }

    int left = max(mb_x * 8 - range, 0);
    int top = max(mb_y * 8 - range, 0);
    int right = min(mb_x * 8 + range, cm->padw[color_component] - 8);
    int bottom = min(mb_y * 8 + range, cm->padh[color_component] - 8);

    int mx = mb_x * 8;
    int my = mb_y * 8;

    // Pointer adjustment for current macroblock
    uint8_t *cu_offset = d_orig + my * cm->padw[color_component] + mx;

    // Launch the kernel
    dim3 blocks((right - left + 15) / 16, (bottom - top + 15) / 16);
    dim3 threads(16, 16);
    computeSADIntegrated<<<blocks, threads>>>(left, right, top, bottom, cm->padw[color_component], mx, my, cu_offset, d_ref, d_best_sad, d_best_x, d_best_y);
}




void c63_motion_estimate(struct c63_common *cm)
{
  int mb_x, mb_y;

  // Initialize device memory before starting motion estimation
  initialize_device_memory(cm);

  // Transfer the original and reconstructed frames to the GPU
  int size_y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT];
  int size_u = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT];
  int size_v = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT];

  cudaMemcpy(d_orig_y, cm->curframe->orig->Y, size_y, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_y, cm->refframe->recons->Y, size_y, cudaMemcpyHostToDevice);

  cudaMemcpy(d_orig_u, cm->curframe->orig->U, size_u, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_u, cm->refframe->recons->U, size_u, cudaMemcpyHostToDevice);

  cudaMemcpy(d_orig_v, cm->curframe->orig->V, size_v, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ref_v, cm->refframe->recons->V, size_v, cudaMemcpyHostToDevice);

/* Luma */
for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
{
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
        me_block_8x8(cm, mb_x, mb_y, Y_COMPONENT, d_orig_y, d_ref_y);
    }
}

/* Chroma */
for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
{
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
        me_block_8x8(cm, mb_x, mb_y, U_COMPONENT, d_orig_u, d_ref_u);
        me_block_8x8(cm, mb_x, mb_y, V_COMPONENT, d_orig_v, d_ref_v);
    }
}


  // Cleanup device memory after motion estimation is complete
  cleanup_device_memory();
}



__global__ void cuda_mc_block_8x8(macroblock mb, int w, uint8_t *predicted, uint8_t *ref, int mv_x, int mv_y)
{
    if (!mb.use_mv) { return; }

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if within bounds to prevent out-of-bounds access
    if (x < 8 && y < 8) {
        predicted[y * w + x] = ref[(y + mv_y) * w + (x + mv_x)];
    }
}



// this had no impact on perfomance... its probably jsut too slow to move the data over... but i tried anyway, and the code looks almost nice
void c63_motion_compensate(struct c63_common *cm)
{
    int mb_x, mb_y;

    // Loop over all macroblocks
    for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
    {
        for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
        {
            // Handle Y, U, and V components separately due to different pointers
            int colors[] = {Y_COMPONENT, U_COMPONENT, V_COMPONENT};
            uint8_t* predicted_ptrs[] = {cm->curframe->predicted->Y, cm->curframe->predicted->U, cm->curframe->predicted->V};
            uint8_t* ref_ptrs[] = {cm->refframe->recons->Y, cm->refframe->recons->U, cm->refframe->recons->V};

            for (int color_idx = 0; color_idx < 3; ++color_idx)
            {
                int color = colors[color_idx];
                int w = cm->padw[color];
                macroblock *mb = &cm->curframe->mbs[color][mb_y * (w/8) + mb_x];
                uint8_t *predicted = predicted_ptrs[color_idx];
                uint8_t *ref = ref_ptrs[color_idx];

                int mv_x = mb->mv_x;
                int mv_y = mb->mv_y;

                // Ensure width and height limits are respected
                if (mb_x < w && mb_y < cm->padh[color])
                {
                    // Launch kernel for the current macroblock and color component
                    cuda_mc_block_8x8<<<1, dim3(8, 8)>>>(*mb, w, predicted, ref, mv_x, mv_y);
                }
            }
        }
    }

    // Synchronize after all kernel launches
    cudaDeviceSynchronize();
}
