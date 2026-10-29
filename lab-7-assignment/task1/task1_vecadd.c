#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void vec_add_serial(const double *a, const double *b, double *c, long n)
{
  for (long i = 0; i < n; i++)
  {
    c[i] = a[i] + b[i];
  }
}

void vec_add_parallel(const double *a, const double *b, double *c, long n, int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    c[i] = a[i] + b[i];
  }
}

int verify_results(const double *c_serial, const double *c_parallel, long n)
{
  for (long i = 0; i < n; i++)
  {
    if (c_serial[i] != c_parallel[i])
    {
      printf("Mismatch at index %ld: Serial = %f, Parallel = %f\n", i, c_serial[i], c_parallel[i]);
      return 0;
    }
  }
  return 1;
}

int main(int argc, char *argv[])
{
  long sizes[] = {100000L, 1000000L, 10000000L, 100000000L}; // N = 10^5, 10^6, 10^7, 10^8

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                  Task 1: Vector Addition Benchmarks                     \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    long n = sizes[s];

    printf("--- Running for N = %ld (%.2f MB total data) ---\n",
           n, (double)(3 * n * sizeof(double)) / (1024 * 1024));

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *c_serial = (double *)malloc(n * sizeof(double));
    double *c_parallel = (double *)malloc(n * sizeof(double));

    if (!a || !b || !c_serial || !c_parallel)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", n);
      return 1;
    }

    for (long i = 0; i < n; i++)
    {
      a[i] = (double)(i * 1.5);
      b[i] = (double)(i * 2.5);
    }

    // Measure Serial Execution Time
    double start_serial = omp_get_wtime();
    vec_add_serial(a, b, c_serial, n);
    double time_serial = omp_get_wtime() - start_serial;

    printf("Serial Execution Time: %8.6f seconds\n", time_serial);
    printf("%-10s | %-15s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
    printf("-----------------------------------------------------------\n");

    // Measure Parallel Execution Time across thread counts
    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_parallel = omp_get_wtime();
      vec_add_parallel(a, b, c_parallel, n, threads);
      double time_parallel = omp_get_wtime() - start_parallel;

      int is_correct = verify_results(c_serial, c_parallel, n);
      double speedup = time_serial / time_parallel;

      printf("%-10d | %-15.6f | %-10.2fx | %-12s\n",
             threads, time_parallel, speedup, is_correct ? "PASSED" : "FAILED");
    }

    free(a);
    free(b);
    free(c_serial);
    free(c_parallel);
    printf("\n");
  }

  return 0;
}