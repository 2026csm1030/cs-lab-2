#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Flattened row-major 1D array indexing: Matrix[i][j] -> Matrix[i * N + j]

// 1. Serial Triple-Loop Implementation (i, j, k order)
void matmul_serial(const double *A, const double *B, double *C, int N)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      double sum = 0.0;
      for (int k = 0; k < N; k++)
      {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// 2. Parallel Outermost Loop Only (#pragma omp parallel for on 'i')
void matmul_parallel_outer(const double *A, const double *B, double *C, int N, int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      double sum = 0.0;

      for (int k = 0; k < N; k++)
      {
        sum += A[i * N + k] * B[k * N + j];
      }

      C[i * N + j] = sum;
    }
  }
}

// 3. Parallel Collapsed Loops (#pragma omp parallel for collapse(2) on 'i' and 'j')
void matmul_parallel_collapse(const double *A, const double *B, double *C, int N, int num_threads)
{
#pragma omp parallel for collapse(2) num_threads(num_threads)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      double sum = 0.0;
      for (int k = 0; k < N; k++)
      {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// 4. Bonus: Loop Order Swapped (i, k, j) - Optimizes spatial locality and cache reuse
void matmul_serial_ikj(const double *A, const double *B, double *C, int N)
{
  for (int i = 0; i < N * N; i++)
    C[i] = 0.0; // Zero initialize C
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      double r = A[i * N + k];
      for (int j = 0; j < N; j++)
      {
        C[i * N + j] += r * B[k * N + j];
      }
    }
  }
}

void matmul_parallel_ikj(const double *A, const double *B, double *C, int N, int num_threads)
{
  for (int i = 0; i < N * N; i++)
  {
    C[i] = 0.0; // Zero initialize C
  }

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      double r = A[i * N + k];
      for (int j = 0; j < N; j++)
      {
        C[i * N + j] += r * B[k * N + j];
      }
    }
  }
}

int verify_results(const double *C_ref, const double *C_test, int N)
{
  double tol = 1e-6;

  for (int i = 0; i < N * N; i++)
  {
    if (fabs(C_ref[i] - C_test[i]) > tol)
    {
      printf("Mismatch at index %d: Ref = %f, Test = %f\n", i, C_ref[i], C_test[i]);
      return 0; // Verification Failed
    }
  }
  return 1; // Verification Passed
}

int main(int argc, char *argv[])
{
  int sizes[] = {256, 512, 1024}; // Matrix dimensions N

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("              Task 5: Matrix Multiplication Benchmarks                   \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    int N = sizes[s];

    long total_elements = (long)N * N;
    double mem_mb = (double)(3 * total_elements * sizeof(double)) / (1024 * 1024);

    printf("=========================================================================\n");
    printf("--- Matrix Size: %d x %d (%.2f MB total data) ---\n", N, N, mem_mb);
    printf("=========================================================================\n");

    double *A = (double *)malloc(total_elements * sizeof(double));
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C_ref = (double *)malloc(total_elements * sizeof(double));
    double *C_test = (double *)malloc(total_elements * sizeof(double));

    if (!A || !B || !C_ref || !C_test)
    {
      fprintf(stderr, "Memory allocation failed for N = %d\n", N);
      return 1;
    }

    // Initialize A and B
    for (long i = 0; i < total_elements; i++)
    {
      A[i] = (double)(i % 100) / 10.0;
      B[i] = (double)(i % 50) / 5.0;
    }

    // 1. Serial Baseline (i-j-k)
    double start_s = omp_get_wtime();
    matmul_serial(A, B, C_ref, N);
    double time_serial = omp_get_wtime() - start_s;

    printf("Serial Baseline Time (i-j-k): %8.6f seconds\n\n", time_serial);

    // 2. Outermost Loop Parallelization (i loop)
    printf("--- Method 1: Outer Loop Parallelization (#pragma omp parallel for) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      matmul_parallel_outer(A, B, C_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(C_ref, C_test, N);
      double speedup = time_serial / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // 3. Collapsed Loops (i and j loops)
    printf("--- Method 2: Collapsed Loops (#pragma omp parallel for collapse(2)) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      matmul_parallel_collapse(A, B, C_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(C_ref, C_test, N);
      double speedup = time_serial / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // 4. Bonus: Optimized Loop Order (i-k-j)
    double start_ikj_s = omp_get_wtime();
    matmul_serial_ikj(A, B, C_test, N);
    double time_ikj_s = omp_get_wtime() - start_ikj_s;

    printf("--- Bonus Method: Swapped Loop Order (i-k-j) ---\n");
    printf("Serial Time (i-k-j): %8.6f seconds (Speedup vs i-j-k baseline: %.2fx)\n",
           time_ikj_s, time_serial / time_ikj_s);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup(self)", "Speedup(base)", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      matmul_parallel_ikj(A, B, C_test, N, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(C_ref, C_test, N);
      double speedup_self = time_ikj_s / time_p;
      double speedup_base = time_serial / time_p;

      printf("%-8d | %-12.6f | %-10.2fx | %-12.2fx | %-12s\n",
             threads, time_p, speedup_self, speedup_base, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    free(A);
    free(B);
    free(C_ref);
    free(C_test);
  }

  return 0;
}