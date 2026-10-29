#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Note: Matrices are stored as flattened contiguous 1D arrays of size N x N
// for optimal memory access and cache performance. C[i][j] -> C[i * N + j].

// 1. Serial Implementation
void mat_add_serial(const double *A, const double *B, double *C, long N)
{
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
  }
}

// 2. Parallel Outer-Loop Only Implementation
void mat_add_parallel_outer(const double *A, const double *B, double *C, long N, int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
  }
}

// 3. Parallel Outer + Inner Collapsed Implementation
void mat_add_parallel_collapse(const double *A, const double *B, double *C, long N, int num_threads)
{
#pragma omp parallel for collapse(2) num_threads(num_threads)
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
  }
}

int verify_matrices(const double *C_ref, const double *C_test, long N)
{
  long total_elements = N * N;
  for (long k = 0; k < total_elements; k++)
  {
    if (C_ref[k] != C_test[k])
    {
      return 0; // Failure
    }
  }
  return 1; // Verification Passed
}

int main(int argc, char *argv[])
{
  long sizes[] = {8L, 4096L}; // Small N = 8, Large N = 4096

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                  Task 4: Matrix Addition Benchmarks                     \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    long N = sizes[s];
    long total_elements = N * N;
    double memory_mb = (double)(3 * total_elements * sizeof(double)) / (1024 * 1024);

    printf("=========================================================================\n");
    printf("--- Matrix Size: %ld x %ld (%ld elements, %.2f MB total data) ---\n",
           N, N, total_elements, memory_mb);
    printf("=========================================================================\n");

    double *A = (double *)malloc(total_elements * sizeof(double));
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C_ref = (double *)malloc(total_elements * sizeof(double));
    double *C_test = (double *)malloc(total_elements * sizeof(double));

    if (!A || !B || !C_ref || !C_test)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", N);
      return 1;
    }

    // Initialize matrices
    for (long k = 0; k < total_elements; k++)
    {
      A[k] = (double)(k * 1.25);
      B[k] = (double)(k * 2.75);
    }

    // Measure Serial Execution
    double start_s = omp_get_wtime();
    mat_add_serial(A, B, C_ref, N);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Execution Time: %8.6f seconds\n\n", time_s);

    // Benchmark Method 1: Outer Loop Only
    printf("--- Method 1: Outer Loop Parallelization (#pragma omp parallel for) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
    printf("----------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      mat_add_parallel_outer(A, B, C_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_matrices(C_ref, C_test, N);
      double speedup = time_s / time_p;

      printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
             threads, time_p, speedup, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Benchmark Method 2: Outer + Inner Collapsed
    printf("--- Method 2: Collapsed Loops (#pragma omp parallel for collapse(2)) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
    printf("----------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      mat_add_parallel_collapse(A, B, C_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_matrices(C_ref, C_test, N);
      double speedup = time_s / time_p;

      printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
             threads, time_p, speedup, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    free(A);
    free(B);
    free(C_ref);
    free(C_test);
  }

  return 0;
}