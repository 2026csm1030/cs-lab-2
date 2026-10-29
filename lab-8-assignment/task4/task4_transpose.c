#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Note: Matrices are stored as 1D contiguous row-major arrays of size N x N.
// Matrix element (row i, col j) is accessed at index [i * N + j].

// 1. Serial Matrix Transpose
void transpose_serial(const double *A, double *B, int N)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B[j * N + i] = A[i * N + j];
    }
  }
}

// 2. Parallel Matrix Transpose (Outer Loop Parallelization)
void transpose_parallel_outer(const double *A, double *B, int N, int num_threads)
{

#pragma omp parallel for num_threads(num_threads)

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B[j * N + i] = A[i * N + j];
    }
  }
}

// 3. Parallel Matrix Transpose with Loop Collapsing
void transpose_parallel_collapse(const double *A, double *B, int N, int num_threads)
{
#pragma omp parallel for collapse(2) num_threads(num_threads)

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B[j * N + i] = A[i * N + j];
    }
  }
}

// 4. Parallel Matrix Transpose with Cache Tiling/Blocking (Optimized Spatial Locality)
void transpose_parallel_tiled(const double *A, double *B, int N, int tile_size, int num_threads)
{
#pragma omp parallel for collapse(2) num_threads(num_threads)

  for (int i = 0; i < N; i += tile_size)
  {
    for (int j = 0; j < N; j += tile_size)
    {
      for (int ii = i; ii < i + tile_size && ii < N; ii++)
      {
        for (int jj = j; jj < j + tile_size && jj < N; jj++)
        {
          B[jj * N + ii] = A[ii * N + jj];
        }
      }
    }
  }
}

// Correctness verification against serial reference
int verify_transpose(const double *B_ref, const double *B_test, int N)
{
  long total_elements = (long)N * N;

  for (long k = 0; k < total_elements; k++)
  {
    if (B_ref[k] != B_test[k])
    {
      return 0; // Mismatch found
    }
  }

  return 1; // Passed
}

int main(int argc, char *argv[])
{
  int sizes[] = {1024, 2048, 4096}; // Matrix sizes N x N

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                  Task 4: Matrix Transpose Benchmarks                    \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    int N = sizes[s];
    long total_elements = (long)N * N;
    double mem_mb = (double)(2 * total_elements * sizeof(double)) / (1024 * 1024);

    printf("=========================================================================\n");
    printf("--- Matrix Size: %d x %d (%.2f MB data) ---\n", N, N, mem_mb);
    printf("=========================================================================\n");

    double *A = (double *)malloc(total_elements * sizeof(double));
    double *B_ref = (double *)malloc(total_elements * sizeof(double));
    double *B_test = (double *)malloc(total_elements * sizeof(double));

    if (!A || !B_ref || !B_test)
    {
      fprintf(stderr, "Memory allocation failed for N = %d\n", N);
      return 1;
    }

    // Initialize matrix A with values
    for (long k = 0; k < total_elements; k++)
    {
      A[k] = (double)(k * 0.75);
    }

    // Serial Execution Baseline
    double start_s = omp_get_wtime();
    transpose_serial(A, B_ref, N);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Transpose Time: %8.6f seconds\n\n", time_s);

    // Method 1: Parallel Outer Loop
    printf("--- Method 1: Outer Loop Parallelization (#pragma omp parallel for) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      transpose_parallel_outer(A, B_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_transpose(B_ref, B_test, N);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 2: Parallel Collapsed Loops
    printf("--- Method 2: Collapsed Loops (#pragma omp parallel for collapse(2)) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      transpose_parallel_collapse(A, B_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_transpose(B_ref, B_test, N);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 3: Parallel Cache Tiling (32x32 tiles)
    printf("--- Method 3: Parallel Tiled Transpose (32x32 Tiles) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      transpose_parallel_tiled(A, B_test, N, 32, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_transpose(B_ref, B_test, N);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }

    free(A);
    free(B_ref);
    free(B_test);
    printf("\n");
  }

  return 0;
}