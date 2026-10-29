#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// 3x3 Averaging Filter Kernel (Sum of elements / 9)
const double KERNEL_3X3[3][3] = {
    {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0},
    {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0},
    {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0}};

// 1. Serial 2D Convolution (Valid padding, output dimension: (H-2) x (W-2))
void conv2d_serial(const double *input, double *output, int H, int W)
{
  int out_H = H - 2;
  int out_W = W - 2;

  for (int i = 0; i < out_H; i++)
  {
    for (int j = 0; j < out_W; j++)
    {
      double sum = 0.0;
      for (int ki = 0; ki < 3; ki++)
      {
        for (int kj = 0; kj < 3; kj++)
        {
          sum += input[(i + ki) * W + (j + kj)] * KERNEL_3X3[ki][kj];
        }
      }
      output[i * out_W + j] = sum;
    }
  }
}

// 2. Parallel 2D Convolution (Outer Loop Parallelization)
void conv2d_parallel_outer(const double *input, double *output, int H, int W, int num_threads)
{
  int out_H = H - 2;
  int out_W = W - 2;

#pragma omp parallel for num_threads(num_threads)

  for (int i = 0; i < out_H; i++)
  {
    for (int j = 0; j < out_W; j++)
    {
      double sum = 0.0;

      for (int ki = 0; ki < 3; ki++)
      {
        for (int kj = 0; kj < 3; kj++)
        {
          sum += input[(i + ki) * W + (j + kj)] * KERNEL_3X3[ki][kj];
        }
      }
      output[i * out_W + j] = sum;
    }
  }
}

// 3. Parallel 2D Convolution (Collapsed Loops)
void conv2d_parallel_collapse(const double *input, double *output, int H, int W, int num_threads)
{
  int out_H = H - 2;
  int out_W = W - 2;

#pragma omp parallel for collapse(2) num_threads(num_threads)
  for (int i = 0; i < out_H; i++)
  {
    for (int j = 0; j < out_W; j++)
    {
      double sum = 0.0;

      for (int ki = 0; ki < 3; ki++)
      {
        for (int kj = 0; kj < 3; kj++)
        {
          sum += input[(i + ki) * W + (j + kj)] * KERNEL_3X3[ki][kj];
        }
      }
      output[i * out_W + j] = sum;
    }
  }
}

// Verification Helper
int verify_conv2d(const double *ref, const double *test, int out_H, int out_W)
{
  double tol = 1e-6;
  long total = (long)out_H * out_W;

  for (long k = 0; k < total; k++)
  {
    if (fabs(ref[k] - test[k]) > tol)
    {
      return 0; // Mismatch
    }
  }
  return 1; // Passed
}

// Example Verification specified in Assignment (5x5 Image -> 3x3 Output)
void run_assignment_example()
{
  printf("=========================================================================\n");
  printf("     Verifying Assignment 5x5 Image Example (3x3 Averaging Kernel)       \n");
  printf("=========================================================================\n");

  int H = 5, W = 5;
  double img_5x5[25] = {
      10, 20, 30, 40, 50,
      60, 70, 80, 90, 100,
      110, 120, 130, 140, 150,
      160, 170, 180, 190, 200,
      210, 220, 230, 240, 250};

  double out_serial[9], out_parallel[9];

  conv2d_serial(img_5x5, out_serial, H, W);
  conv2d_parallel_outer(img_5x5, out_parallel, H, W, 4);

  printf("Serial Output (3x3 Matrix):\n");
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      printf("%7.2f ", out_serial[i * 3 + j]);
    }
    printf("\n");
  }

  printf("\nParallel Output (3x3 Matrix, 4 Threads):\n");
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      printf("%7.2f ", out_parallel[i * 3 + j]);
    }
    printf("\n");
  }

  int ok = verify_conv2d(out_serial, out_parallel, 3, 3);
  printf("Sample Position O[1][1] Expected = 70.00, Computed = %.2f\n", out_serial[0]);
  printf("Assignment Example Status: %s\n\n", ok ? "PASSED" : "FAILED");
}

int main(int argc, char *argv[])
{
  // Run the specified assignment 5x5 example
  run_assignment_example();

  int sizes[] = {1024, 2048, 4096}; // Image Dimensions (N x N)

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                 Task 5: 2D Convolution Benchmarks                       \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    int N = sizes[s];
    int H = N, W = N;
    int out_H = H - 2, out_W = W - 2;

    long in_elems = (long)H * W;
    long out_elems = (long)out_H * out_W;
    double mem_mb = (double)((in_elems + 2 * out_elems) * sizeof(double)) / (1024 * 1024);

    printf("=========================================================================\n");
    printf("--- Image Size: %d x %d (Output: %d x %d, %.2f MB data) ---\n",
           H, W, out_H, out_W, mem_mb);
    printf("=========================================================================\n");

    double *input = (double *)malloc(in_elems * sizeof(double));
    double *output_ref = (double *)malloc(out_elems * sizeof(double));
    double *output_test = (double *)malloc(out_elems * sizeof(double));

    if (!input || !output_ref || !output_test)
    {
      fprintf(stderr, "Memory allocation failed for N = %d\n", N);
      return 1;
    }

    // Initialize Image Matrix
    for (long k = 0; k < in_elems; k++)
    {
      input[k] = (double)(k % 256);
    }

    // Serial Baseline
    double start_s = omp_get_wtime();
    conv2d_serial(input, output_ref, H, W);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial 2D Conv Time: %8.6f seconds\n\n", time_s);

    // Method 1: Outer Loop Parallelization
    printf("--- Method 1: Outer Loop Parallelization (#pragma omp parallel for) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      conv2d_parallel_outer(input, output_test, H, W, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_conv2d(output_ref, output_test, out_H, out_W);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 2: Collapsed Loops
    printf("--- Method 2: Collapsed Loops (#pragma omp parallel for collapse(2)) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      conv2d_parallel_collapse(input, output_test, H, W, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_conv2d(output_ref, output_test, out_H, out_W);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }

    free(input);
    free(output_ref);
    free(output_test);
    printf("\n");
  }

  return 0;
}