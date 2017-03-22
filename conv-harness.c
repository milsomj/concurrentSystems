/* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   February 2017


   Version 1.4 : Modified the random generator to reduce the range
                 of generated values;
                 Changed the summation in the checking code from
                 float to double to try to bring the checked value
                 closer to the "true" value

   Version 1.3 : Fixed which loop variables were being incremented
                 in write_out();
                 Fixed dimensions of output and control_output 
                 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully) 
                 eliminate random walk of floating point error;
                 Also introduced checks to restrict kernel-order to
                 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <x86intrin.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write 3d matrix to stdout */
void write_out(float *** a, int dim0, int dim1, int dim2)
{
  int i, j, k;

  for ( i = 0; i < dim0; i++ ) {
    printf("Outer dimension number %d\n", i);
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2 - 1; k++ ) {
        printf("%f, ", a[i][j][k]);
      }
      // print end of line
      printf("%f\n", a[i][j][dim2-1]);
    }
  }
}


/* create new empty 4d matrix */
float **** new_empty_4d_matrix(int dim0, int dim1, int dim2, int dim3)
{
  float **** result = malloc(dim0 * sizeof(float***));
  float *** mat1 = malloc(dim0 * dim1 * sizeof(float**));
  float ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float*));
  float * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(float));
  int i, j, k;

  
  for ( i = 0; i < dim0; i++ ) {
    result[i] = &(mat1[i*dim1]);
    for ( j = 0; j < dim1; j++ ) {
      result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
      for ( k = 0; k < dim2; k++ ) {
        result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
float *** new_empty_3d_matrix(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
float **** copy_4d_matrix(float **** source_matrix, int dim0,
                            int dim1, int dim2, int dim3)
{
  int i, j, k, l;
  float **** result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);

  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          result[i][j][k][l] = source_matrix[i][j][k][l];
        }
      }
    }
  }
  return result;
}

/* create a matrix and fill it with random numbers */
float **** gen_random_4d_matrix(int dim0, int dim1, int dim2, int dim3)
{
  float **** result;
  int i, j, k, l;
  struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 12; // 2^12
  const int bias = 1 << 16; // 2^16
  float offset = 0.0;
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          float num = (((float) reduced_range) / ((float) bias))+offset;
          result[i][j][k][l] = num;
        }
      }
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
float *** gen_random_3d_matrix(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(float *** result, float *** control,
                  int dim0, int dim1, int dim2)
{
  int i, j, k;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  //printf("SAD\n");
  
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        double diff = fabs(control[i][j][k] - result[i][j][k]);
        assert( diff >= 0.0 );
        sum_abs_diff = sum_abs_diff + diff;
      }
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
  else {
    printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
  }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(float *** image, float **** kernels, float *** output,
                       int width, int height, int nchannels, int nkernels,
                       int kernel_order)
{
  int h, w, x, y, c, m;

  for ( m = 0; m < nkernels; m++ ) {
    for ( w = 0; w < width; w++ ) {
      for ( h = 0; h < height; h++ ) {
        double sum = 0.0;
        for ( c = 0; c < nchannels; c++ ) {
          for ( x = 0; x < kernel_order; x++) {
            for ( y = 0; y < kernel_order; y++ ) {
              sum += image[w+x][h+y][c] * kernels[m][c][x][y];
            }
          }
          output[m][w][h] = sum;
        }
      }
    }
  }
}

/* the fast version of matmul written by the team */
void team_conv(float *** image, float **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int h, w, x, y, c, m;
  float **** kerns = new_empty_4d_matrix(nkernels,kernel_order,kernel_order,nchannels);
  int i,j,k,l;
  __m128 a4,b4,c4,sum4;
  float temp[] = {0.0,0.0,0.0,0.0};
  if(nchannels > 3){
    switch(kernel_order){
      case 1:
        for(i = 0; i< nkernels;i++){
          for(j = 0; j< nchannels;j++){
            for(k = 0; k< kernel_order;k++){
              for(l = 0; l< kernel_order;l++){
                kerns[i][k][l][j] = kernels[i][j][k][l];
              }
            }
          } 
        }
        #pragma omp parallel for private(m,w,h,c,a4,b4,c4,sum4,temp,i) collapse(3) 
        for ( m = 0; m < nkernels; m++ ) {
          for ( w = 0; w < width; w++ ) {
            for ( h = 0; h < height; h++ ) {
              double sum;
              sum4 = _mm_setzero_ps();
              for ( c = 0; c < nchannels-3; c+=4 ) {
                a4 = _mm_loadu_ps(&image[w][h][c]);
                b4 = _mm_loadu_ps(&kerns[m][0][0][c]);
                c4 = _mm_mul_ps(a4,b4);
                sum4 = _mm_add_ps(sum4,c4);
              }
              _mm_storeu_ps(&temp[0],sum4);
              sum = temp[0] + temp[1] + temp[2] + temp[3];

              for(i = c; i < nchannels; i++){
                sum += image[w][h][i] * kerns[m][0][0][i];
              }
              output[m][w][h] = sum;
            }
          }
        }
        break;
      case 3:
        for(i = 0; i< nkernels;i++){
          for(j = 0; j< nchannels;j++){
            for(k = 0; k< kernel_order;k++){
              for(l = 0; l< kernel_order;l++){
                kerns[i][k][l][j] = kernels[i][j][k][l];
              }
            }
          } 
        }
        #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
        for ( m = 0; m < nkernels; m++ ) {
          for ( w = 0; w < width; w++ ) {
            for ( h = 0; h < height; h++ ) {
              double sum;
              sum4 = _mm_setzero_ps();
              for ( c = 0; c < nchannels-3; c+=4 ) {
                a4=_mm_loadu_ps(&image[w+0][h+0][c]); b4=_mm_loadu_ps(&kerns[m][0][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+1][c]); b4=_mm_loadu_ps(&kerns[m][0][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+2][c]); b4=_mm_loadu_ps(&kerns[m][0][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+0][c]); b4=_mm_loadu_ps(&kerns[m][1][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+1][c]); b4=_mm_loadu_ps(&kerns[m][1][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+2][c]); b4=_mm_loadu_ps(&kerns[m][1][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+0][c]); b4=_mm_loadu_ps(&kerns[m][2][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+1][c]); b4=_mm_loadu_ps(&kerns[m][2][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+2][c]); b4=_mm_loadu_ps(&kerns[m][2][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);
              }
              _mm_storeu_ps(&temp[0],sum4);
              sum = temp[0] + temp[1] + temp[2] + temp[3];
              for ( x = 0; x < kernel_order; x++) {
                for ( y = 0; y < kernel_order; y++ ) {
                  for(i = c; i < nchannels; i++){
                    sum += image[w+x][h+y][i] * kerns[m][x][y][i];
                  }
                }
              }
              output[m][w][h] = sum;
            }
          }
        }
        break;
      case 5:
        for(i = 0; i< nkernels;i++){
          for(j = 0; j< nchannels;j++){
            for(k = 0; k< kernel_order;k++){
              for(l = 0; l< kernel_order;l++){
                kerns[i][k][l][j] = kernels[i][j][k][l];
              }
            }
          } 
        }
        #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
        for ( m = 0; m < nkernels; m++ ) {
          for ( w = 0; w < width; w++ ) {
            for ( h = 0; h < height; h++ ) {
              double sum;
              sum4 = _mm_setzero_ps();
              for ( c = 0; c < nchannels-3; c+=4 ) {
                a4=_mm_loadu_ps(&image[w+0][h+0][c]); b4=_mm_loadu_ps(&kerns[m][0][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+1][c]); b4=_mm_loadu_ps(&kerns[m][0][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+2][c]); b4=_mm_loadu_ps(&kerns[m][0][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+3][c]); b4=_mm_loadu_ps(&kerns[m][0][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+4][c]); b4=_mm_loadu_ps(&kerns[m][0][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+0][c]); b4=_mm_loadu_ps(&kerns[m][1][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+1][c]); b4=_mm_loadu_ps(&kerns[m][1][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+2][c]); b4=_mm_loadu_ps(&kerns[m][1][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+3][c]); b4=_mm_loadu_ps(&kerns[m][1][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+4][c]); b4=_mm_loadu_ps(&kerns[m][1][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+0][c]); b4=_mm_loadu_ps(&kerns[m][2][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+1][c]); b4=_mm_loadu_ps(&kerns[m][2][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+2][c]); b4=_mm_loadu_ps(&kerns[m][2][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+3][c]); b4=_mm_loadu_ps(&kerns[m][2][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+4][c]); b4=_mm_loadu_ps(&kerns[m][2][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+0][c]); b4=_mm_loadu_ps(&kerns[m][3][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+1][c]); b4=_mm_loadu_ps(&kerns[m][3][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+2][c]); b4=_mm_loadu_ps(&kerns[m][3][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+3][c]); b4=_mm_loadu_ps(&kerns[m][3][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+4][c]); b4=_mm_loadu_ps(&kerns[m][3][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+0][c]); b4=_mm_loadu_ps(&kerns[m][4][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+1][c]); b4=_mm_loadu_ps(&kerns[m][4][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+2][c]); b4=_mm_loadu_ps(&kerns[m][4][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+3][c]); b4=_mm_loadu_ps(&kerns[m][4][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+4][c]); b4=_mm_loadu_ps(&kerns[m][4][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);
              }
              _mm_storeu_ps(&temp[0],sum4);
              sum = temp[0] + temp[1] + temp[2] + temp[3];
              for ( x = 0; x < kernel_order; x++) {
                for ( y = 0; y < kernel_order; y++ ) {
                  for(i = c; i < nchannels; i++){
                    sum += image[w+x][h+y][i] * kerns[m][x][y][i];
                  }
                }
              }
              output[m][w][h] = sum;
            }
          }
        }
        break;
      case 7:
        for(i = 0; i< nkernels;i++){
          for(j = 0; j< nchannels;j++){
            for(k = 0; k< kernel_order;k++){
              for(l = 0; l< kernel_order;l++){
                kerns[i][k][l][j] = kernels[i][j][k][l];
              }
            }
          } 
        }
        #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) //schedule(dynamic)
        for ( m = 0; m < nkernels; m++ ) {
          for ( w = 0; w < width; w++ ) {
            for ( h = 0; h < height; h++ ) {
              double sum;
              sum4 = _mm_setzero_ps();
              for ( c = 0; c < nchannels-3; c+=4 ) {
                a4=_mm_loadu_ps(&image[w+0][h+0][c]); b4=_mm_loadu_ps(&kerns[m][0][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+1][c]); b4=_mm_loadu_ps(&kerns[m][0][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+2][c]); b4=_mm_loadu_ps(&kerns[m][0][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+3][c]); b4=_mm_loadu_ps(&kerns[m][0][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+4][c]); b4=_mm_loadu_ps(&kerns[m][0][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+5][c]); b4=_mm_loadu_ps(&kerns[m][0][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+0][h+6][c]); b4=_mm_loadu_ps(&kerns[m][0][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+0][c]); b4=_mm_loadu_ps(&kerns[m][1][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+1][c]); b4=_mm_loadu_ps(&kerns[m][1][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+2][c]); b4=_mm_loadu_ps(&kerns[m][1][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+3][c]); b4=_mm_loadu_ps(&kerns[m][1][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+4][c]); b4=_mm_loadu_ps(&kerns[m][1][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+5][c]); b4=_mm_loadu_ps(&kerns[m][1][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+1][h+6][c]); b4=_mm_loadu_ps(&kerns[m][1][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+0][c]); b4=_mm_loadu_ps(&kerns[m][2][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+1][c]); b4=_mm_loadu_ps(&kerns[m][2][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+2][c]); b4=_mm_loadu_ps(&kerns[m][2][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+3][c]); b4=_mm_loadu_ps(&kerns[m][2][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+4][c]); b4=_mm_loadu_ps(&kerns[m][2][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+5][c]); b4=_mm_loadu_ps(&kerns[m][2][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+2][h+6][c]); b4=_mm_loadu_ps(&kerns[m][2][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+0][c]); b4=_mm_loadu_ps(&kerns[m][3][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+1][c]); b4=_mm_loadu_ps(&kerns[m][3][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+2][c]); b4=_mm_loadu_ps(&kerns[m][3][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+3][c]); b4=_mm_loadu_ps(&kerns[m][3][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+4][c]); b4=_mm_loadu_ps(&kerns[m][3][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+5][c]); b4=_mm_loadu_ps(&kerns[m][3][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+3][h+6][c]); b4=_mm_loadu_ps(&kerns[m][3][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+0][c]); b4=_mm_loadu_ps(&kerns[m][4][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+1][c]); b4=_mm_loadu_ps(&kerns[m][4][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+2][c]); b4=_mm_loadu_ps(&kerns[m][4][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+3][c]); b4=_mm_loadu_ps(&kerns[m][4][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+4][c]); b4=_mm_loadu_ps(&kerns[m][4][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+5][c]); b4=_mm_loadu_ps(&kerns[m][4][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+4][h+6][c]); b4=_mm_loadu_ps(&kerns[m][4][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+0][c]); b4=_mm_loadu_ps(&kerns[m][5][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+1][c]); b4=_mm_loadu_ps(&kerns[m][5][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+2][c]); b4=_mm_loadu_ps(&kerns[m][5][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+3][c]); b4=_mm_loadu_ps(&kerns[m][5][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+4][c]); b4=_mm_loadu_ps(&kerns[m][5][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+5][c]); b4=_mm_loadu_ps(&kerns[m][5][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+5][h+6][c]); b4=_mm_loadu_ps(&kerns[m][5][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+0][c]); b4=_mm_loadu_ps(&kerns[m][6][0][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+1][c]); b4=_mm_loadu_ps(&kerns[m][6][1][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+2][c]); b4=_mm_loadu_ps(&kerns[m][6][2][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+3][c]); b4=_mm_loadu_ps(&kerns[m][6][3][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+4][c]); b4=_mm_loadu_ps(&kerns[m][6][4][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+5][c]); b4=_mm_loadu_ps(&kerns[m][6][5][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);a4=_mm_loadu_ps(&image[w+6][h+6][c]); b4=_mm_loadu_ps(&kerns[m][6][6][c]); c4=_mm_mul_ps(a4,b4); sum4=_mm_add_ps(sum4,c4);
              }
              _mm_storeu_ps(&temp[0],sum4);
              sum = temp[0] + temp[1] + temp[2] + temp[3];
              for ( x = 0; x < kernel_order; x++) {
                for ( y = 0; y < kernel_order; y++ ) {
                  for(i = c; i < nchannels; i++){
                    sum += image[w+x][h+y][i] * kerns[m][x][y][i];
                  }
                }
              }
              output[m][w][h] = sum;
            }
          }
        }
    } 
  }
  else{
    switch(kernel_order){
    case 1:
      for(i = 0; i< nkernels;i++){
         for(j = 0; j< nchannels;j++){
           for(k = 0; k< kernel_order;k++){
             for(l = 0; l< kernel_order;l++){
               kerns[i][k][l][j] = kernels[i][j][k][l];
             }
           }
         } 
       }
      #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
      for ( m = 0; m < nkernels; m++ ) {
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            for ( c = 0; c < nchannels; c++ ) {
              sum += image[w][h][c] * kernels[m][c][0][0];
            }   
            output[m][w][h] = sum;
          }
        }
      }
      break;
    case 3:
      for(i = 0; i< nkernels;i++){
         for(j = 0; j< nchannels;j++){
           for(k = 0; k< kernel_order;k++){
             for(l = 0; l< kernel_order;l++){
               kerns[i][k][l][j] = kernels[i][j][k][l];
             }
           }
         } 
       }
      #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
      for ( m = 0; m < nkernels; m++ ) {
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            for ( c = 0; c < nchannels; c++ ) {
              sum += (image[w][h][c] * kernels[m][c][0][0]) + (image[w][h+1][c] * kernels[m][c][0][1]) + (image[w][h+2][c] * kernels[m][c][0][2]) + (image[w+1][h][c] * kernels[m][c][1][0]) + (image[w+1][h+1][c] * kernels[m][c][1][1]) + (image[w+1][h+2][c] * kernels[m][c][1][2]) + (image[w+2][h][c] * kernels[m][c][2][0]) + (image[w+2][h+1][c] * kernels[m][c][2][1]) + (image[w+2][h+2][c] * kernels[m][c][2][2]);
            }   
            output[m][w][h] = sum;
          }
        }
      }
      break;
    case 5:
      for(i = 0; i< nkernels;i++){
         for(j = 0; j< nchannels;j++){
           for(k = 0; k< kernel_order;k++){
             for(l = 0; l< kernel_order;l++){
               kerns[i][k][l][j] = kernels[i][j][k][l];
             }
           }
         } 
       }
      #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
      for ( m = 0; m < nkernels; m++ ) {
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            for ( c = 0; c < nchannels; c++ ) {
              sum += (image[w+0][h+0][c] * kernels[m][c][0][0]) + (image[w+0][h+1][c] * kernels[m][c][0][1]) + (image[w+0][h+2][c] * kernels[m][c][0][2]) + (image[w+0][h+3][c] * kernels[m][c][0][3]) + (image[w+0][h+4][c] * kernels[m][c][0][4]) + (image[w+1][h+0][c] * kernels[m][c][1][0]) + (image[w+1][h+1][c] * kernels[m][c][1][1]) + (image[w+1][h+2][c] * kernels[m][c][1][2]) + (image[w+1][h+3][c] * kernels[m][c][1][3]) + (image[w+1][h+4][c] * kernels[m][c][1][4]) + (image[w+2][h+0][c] * kernels[m][c][2][0]) + (image[w+2][h+1][c] * kernels[m][c][2][1]) + (image[w+2][h+2][c] * kernels[m][c][2][2]) + (image[w+2][h+3][c] * kernels[m][c][2][3]) + (image[w+2][h+4][c] * kernels[m][c][2][4]) + (image[w+3][h+0][c] * kernels[m][c][3][0]) + (image[w+3][h+1][c] * kernels[m][c][3][1]) + (image[w+3][h+2][c] * kernels[m][c][3][2]) + (image[w+3][h+3][c] * kernels[m][c][3][3]) + (image[w+3][h+4][c] * kernels[m][c][3][4]) + (image[w+4][h+0][c] * kernels[m][c][4][0]) + (image[w+4][h+1][c] * kernels[m][c][4][1]) + (image[w+4][h+2][c] * kernels[m][c][4][2]) + (image[w+4][h+3][c] * kernels[m][c][4][3]) + (image[w+4][h+4][c] * kernels[m][c][4][4]);
            }   
            output[m][w][h] = sum;
          }
        }
      }
      break;
    case 7:
      for(i = 0; i< nkernels;i++){
         for(j = 0; j< nchannels;j++){
           for(k = 0; k< kernel_order;k++){
             for(l = 0; l< kernel_order;l++){
               kerns[i][k][l][j] = kernels[i][j][k][l];
             }
           }
         } 
       }
      #pragma omp parallel for private(m,w,h,c,x,y,a4,b4,c4,sum4,temp,i) collapse(3) 
      for ( m = 0; m < nkernels; m++ ) {
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            for ( c = 0; c < nchannels; c++ ) {
              sum += (image[w+0][h+0][c] * kernels[m][c][0][0]) + (image[w+0][h+1][c] * kernels[m][c][0][1]) + (image[w+0][h+2][c] * kernels[m][c][0][2]) + (image[w+0][h+3][c] * kernels[m][c][0][3]) + (image[w+0][h+4][c] * kernels[m][c][0][4]) + (image[w+0][h+5][c] * kernels[m][c][0][5]) + (image[w+0][h+6][c] * kernels[m][c][0][6]) + (image[w+1][h+0][c] * kernels[m][c][1][0]) + (image[w+1][h+1][c] * kernels[m][c][1][1]) + (image[w+1][h+2][c] * kernels[m][c][1][2]) + (image[w+1][h+3][c] * kernels[m][c][1][3]) + (image[w+1][h+4][c] * kernels[m][c][1][4]) + (image[w+1][h+5][c] * kernels[m][c][1][5]) + (image[w+1][h+6][c] * kernels[m][c][1][6]) + (image[w+2][h+0][c] * kernels[m][c][2][0]) + (image[w+2][h+1][c] * kernels[m][c][2][1]) + (image[w+2][h+2][c] * kernels[m][c][2][2]) + (image[w+2][h+3][c] * kernels[m][c][2][3]) + (image[w+2][h+4][c] * kernels[m][c][2][4]) + (image[w+2][h+5][c] * kernels[m][c][2][5]) + (image[w+2][h+6][c] * kernels[m][c][2][6]) + (image[w+3][h+0][c] * kernels[m][c][3][0]) + (image[w+3][h+1][c] * kernels[m][c][3][1]) + (image[w+3][h+2][c] * kernels[m][c][3][2]) + (image[w+3][h+3][c] * kernels[m][c][3][3]) + (image[w+3][h+4][c] * kernels[m][c][3][4]) + (image[w+3][h+5][c] * kernels[m][c][3][5]) + (image[w+3][h+6][c] * kernels[m][c][3][6]) + (image[w+4][h+0][c] * kernels[m][c][4][0]) + (image[w+4][h+1][c] * kernels[m][c][4][1]) + (image[w+4][h+2][c] * kernels[m][c][4][2]) + (image[w+4][h+3][c] * kernels[m][c][4][3]) + (image[w+4][h+4][c] * kernels[m][c][4][4]) + (image[w+4][h+5][c] * kernels[m][c][4][5]) + (image[w+4][h+6][c] * kernels[m][c][4][6]) + (image[w+5][h+0][c] * kernels[m][c][5][0]) + (image[w+5][h+1][c] * kernels[m][c][5][1]) + (image[w+5][h+2][c] * kernels[m][c][5][2]) + (image[w+5][h+3][c] * kernels[m][c][5][3]) + (image[w+5][h+4][c] * kernels[m][c][5][4]) + (image[w+5][h+5][c] * kernels[m][c][5][5]) + (image[w+5][h+6][c] * kernels[m][c][5][6]) + (image[w+6][h+0][c] * kernels[m][c][6][0]) + (image[w+6][h+1][c] * kernels[m][c][6][1]) + (image[w+6][h+2][c] * kernels[m][c][6][2]) + (image[w+6][h+3][c] * kernels[m][c][6][3]) + (image[w+6][h+4][c] * kernels[m][c][6][4]) + (image[w+6][h+5][c] * kernels[m][c][6][5]) + (image[w+6][h+6][c] * kernels[m][c][6][6]);
            }   
            output[m][w][h] = sum;
          }
        }
      }
    }
  }
}

int main(int argc, char ** argv)
{
  //float image[W][H][C];
  //float kernels[M][C][K][K];
  //float output[M][W][H];
  
  float *** image, **** kernels, *** output;
  float *** control_output;
  long long mul_time;
  int width, height, kernel_order, nchannels, nkernels;
  struct timeval start_time1;
  struct timeval stop_time1;
  //Added variables for timer
  long long mul_time1, mul_time2;
  struct timeval start_time2;
  struct timeval stop_time2;

  if ( argc != 6 ) {
    fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
    exit(1);
  }
  else {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    kernel_order = atoi(argv[3]);
    nchannels = atoi(argv[4]);
    nkernels = atoi(argv[5]);
  }
  switch ( kernel_order ) {
  case 1:
  case 3:
  case 5:
  case 7: break;
  default:
    fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
            kernel_order);
    exit(1);
  }

  /* allocate the matrices */
  image = gen_random_3d_matrix(width+kernel_order, height + kernel_order,
                               nchannels);
  kernels = gen_random_4d_matrix(nkernels, nchannels, kernel_order, kernel_order);
  output = new_empty_3d_matrix(nkernels, width, height);
  control_output = new_empty_3d_matrix(nkernels, width, height);

  //DEBUGGING(write_out(A, a_dim1, a_dim2));

  gettimeofday(&start_time1, NULL);
  /* use a simple multichannel convolution routine to produce control result */
  multichannel_conv(image, kernels, control_output, width,
                    height, nchannels, nkernels, kernel_order);
  gettimeofday(&stop_time1, NULL);

  /* record starting time of team's code*/
  gettimeofday(&start_time2, NULL);

  /* perform student team's multichannel convolution */
  team_conv(image, kernels, output, width,
                    height, nchannels, nkernels, kernel_order);

  /* record finishing time */
  gettimeofday(&stop_time2, NULL);
  mul_time1 = (stop_time1.tv_sec - start_time1.tv_sec) * 1000000L +
    (stop_time1.tv_usec - start_time1.tv_usec);
  mul_time2 = (stop_time2.tv_sec - start_time2.tv_sec) * 1000000L +
    (stop_time2.tv_usec - start_time2.tv_usec);
  printf("     Dave time: %lld microseconds  Team conv time: %lld microseconds\n", mul_time1, mul_time2);
  printf("         Optimisation Factor: %lld\n", mul_time1/mul_time2);

  DEBUGGING(write_out(output, nkernels, width, height));

  /* now check that the team's multichannel convolution routine
     gives the same answer as the known working version */
  check_result(output, control_output, nkernels, width, height);

  return 0;
}
