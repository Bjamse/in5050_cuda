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
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }


  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;


  uint8_t *cu_orig, *cu_ref;


  cudaMemcpy(cu_orig, orig, w*h * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cu_ref, ref, w*h * sizeof(uint8_t), cudaMemcpyHostToDevice);


  uint8_t * cu_offsett = cu_orig + my*w+mx;
  // Define variables for minimum SAD and motion vectors

  int best_x = 0, best_y = 0;

  // Allocate memory for best_sad, best_x, best_y on device
  int *d_best_sad, *d_best_x, *d_best_y;
  cudaMalloc(&d_best_sad, sizeof(int));
  cudaMalloc(&d_best_x, sizeof(int));
  cudaMalloc(&d_best_y, sizeof(int));
  cudaMemcpy(d_best_sad, &best_sad, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_best_x, &best_x, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_best_y, &best_y, sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  dim3 blocks((right - left + 15) / 16, (bottom - top + 15) / 16);
  dim3 threads(16, 16);

  // Launch the kernel
  computeSADIntegrated<<<blocks, threads>>>(left, right, top, bottom, w, mx, my, cu_offsett, cu_ref, d_best_sad, d_best_x, d_best_y);

  // Copy results back to host
  cudaMemcpy(&best_sad, d_best_sad, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&best_x, d_best_x, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&best_y, d_best_y, sizeof(int), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_best_sad);
  cudaFree(d_best_x);
  cudaFree(d_best_y);

  cudaFree(cu_orig);
  cudaFree(cu_ref);

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}






void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
