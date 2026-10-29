#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Scaled workload factor to allow N = 10^7 to finish quickly
static inline double compute_uneven_work(long i, double val)
{
  int work = (int)((i % 1000) * 5); // Scaled multiplier for fast execution
  double x = val;

  for (int j = 0; j < work; j++)
  {
    x = x * 1.000001 + 0.0000001;
  }

  return x;
}

// 1. Serial Implementation Baseline
void run_serial(double *a, long n)
{
  for (long i = 0; i < n; i++)
  {
    a[i] = compute_uneven_work(i, a[i]);
  }
}

// 2. Parallel static schedule
void run_parallel_static(double *a, long n, int num_threads)
{
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    a[i] = compute_uneven_work(i, a[i]);
  }
}

// 3. Parallel dynamic schedule (chunk size = 100)
void run_parallel_dynamic(double *a, long n, int chunk_size, int num_threads)
{
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    a[i] = compute_uneven_work(i, a[i]);
  }
}

// 4. Parallel guided schedule
void run_parallel_guided(double *a, long n, int num_threads)
{
#pragma omp parallel for schedule(guided) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    a[i] = compute_uneven_work(i, a[i]);
  }
}

int verify_results(const double *a_ref, const double *a_test, long n)
{
  double tol = 1e-5;

  for (long i = 0; i < n; i++)
  {
    if (fabs(a_ref[i] - a_test[i]) > tol)
    {
      return 0; // Verification Failed
    }
  }

  return 1; // Verification Passed
}

int main(int argc, char *argv[])
{
  long sizes[] = {100000L, 1000000L, 10000000L}; // N = 10^5, 10^6, 10^7

  int thread_counts[] = {1, 2, 4, 8};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);
  int chunk_size = 100;

  printf("=========================================================================\n");
  printf("        Task 2: OpenMP Scheduling Strategies (Uneven Workloads)         \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    long n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Running Benchmarks for N = %ld (%.2f MB array) ---\n",
           n, (double)(n * sizeof(double)) / (1024 * 1024));
    printf("=========================================================================\n");

    double *a_orig = (double *)malloc(n * sizeof(double));
    double *a_ref = (double *)malloc(n * sizeof(double));
    double *a_test = (double *)malloc(n * sizeof(double));

    if (!a_orig || !a_ref || !a_test)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", n);
      return 1;
    }

    for (long i = 0; i < n; i++)
    {
      a_orig[i] = i * 0.001;
      a_ref[i] = a_orig[i];
    }

    // Serial Baseline
    double start_s = omp_get_wtime();
    run_serial(a_ref, n);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Execution Time: %8.6f seconds\n\n", time_s);

    // Schedule Static
    printf("--- Schedule: static ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      for (long i = 0; i < n; i++)
        a_test[i] = a_orig[i];

      double start_p = omp_get_wtime();
      run_parallel_static(a_test, n, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(a_ref, a_test, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Schedule Dynamic (chunk_size = 100)
    printf("--- Schedule: dynamic, %d ---\n", chunk_size);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      for (long i = 0; i < n; i++)
      {
        a_test[i] = a_orig[i];
      }

      double start_p = omp_get_wtime();
      run_parallel_dynamic(a_test, n, chunk_size, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(a_ref, a_test, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Schedule Guided
    printf("--- Schedule: guided ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      for (long i = 0; i < n; i++)
      {
        a_test[i] = a_orig[i];
      }

      double start_p = omp_get_wtime();
      run_parallel_guided(a_test, n, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(a_ref, a_test, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }

    free(a_orig);
    free(a_ref);
    free(a_test);
    printf("\n");
  }

  return 0;
}