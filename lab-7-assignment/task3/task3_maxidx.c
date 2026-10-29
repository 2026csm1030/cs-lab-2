#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#define MAX_THREADS 128

// 1. Serial Implementation (Max & Min with Indexes)
void find_max_min_serial(const int *a, long n, int *max_val, long *max_idx, int *min_val, long *min_idx)
{
  int max_v = a[0], min_v = a[0];
  long max_i = 0, min_i = 0;

  for (long i = 1; i < n; i++)
  {
    if (a[i] > max_v)
    {
      max_v = a[i];
      max_i = i;
    }
    if (a[i] < min_v)
    {
      min_v = a[i];
      min_i = i;
    }
  }

  *max_val = max_v;
  *max_idx = max_i;
  *min_val = min_v;
  *min_idx = min_i;
}

// 2. OpenMP Value-only Reduction for Max/Min
void find_max_min_reduction(const int *a, long n, int *max_val, int *min_val, int num_threads)
{
  int max_v = INT_MIN;
  int min_v = INT_MAX;

#pragma omp parallel for reduction(max : max_v) reduction(min : min_v) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    if (a[i] > max_v)
      max_v = a[i];
    if (a[i] < min_v)
      min_v = a[i];
  }

  *max_val = max_v;
  *min_val = min_v;
}

// 3. Critical-Section Approach for (max_val, max_idx)
void find_max_idx_critical(const int *a, long n, int *max_val, long *max_idx, int num_threads)
{
  int best_val = a[0];
  long best_idx = 0;

#pragma omp parallel for num_threads(num_threads)
  for (long i = 1; i < n; i++)
  {
#pragma omp critical
    {
      if (a[i] > best_val)
      {
        best_val = a[i];
        best_idx = i;
      }
    }
  }

  *max_val = best_val;
  *max_idx = best_idx;
}

// 4. Manual Per-Thread Reduction Approach for (max_val, max_idx) & (min_val, min_idx)
void find_max_min_idx_manual(const int *a, long n, int *max_val, long *max_idx,
                             int *min_val, long *min_idx, int num_threads)
{
  int local_max_val[MAX_THREADS];
  long local_max_idx[MAX_THREADS];
  int local_min_val[MAX_THREADS];
  long local_min_idx[MAX_THREADS];

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int my_max_v = INT_MIN, my_min_v = INT_MAX;
    long my_max_i = -1, my_min_i = -1;

#pragma omp for
    for (long i = 0; i < n; i++)
    {
      if (a[i] > my_max_v)
      {
        my_max_v = a[i];
        my_max_i = i;
      }
      if (a[i] < my_min_v)
      {
        my_min_v = a[i];
        my_min_i = i;
      }
    }

    local_max_val[tid] = my_max_v;
    local_max_idx[tid] = my_max_i;
    local_min_val[tid] = my_min_v;
    local_min_idx[tid] = my_min_i;
  }

  // Serial Pass to combine per-thread results
  int global_max_v = local_max_val[0];
  long global_max_i = local_max_idx[0];
  int global_min_v = local_min_val[0];
  long global_min_i = local_min_idx[0];

  for (int t = 1; t < num_threads; t++)
  {
    if (local_max_val[t] > global_max_v)
    {
      global_max_v = local_max_val[t];
      global_max_i = local_max_idx[t];
    }
    if (local_min_val[t] < global_min_v)
    {
      global_min_v = local_min_val[t];
      global_min_i = local_min_idx[t];
    }
  }

  *max_val = global_max_v;
  *max_idx = global_max_i;
  *min_val = global_min_v;
  *min_idx = global_min_i;
}

int main(int argc, char *argv[])
{
  long sizes[] = {10000000L, 100000000L}; // N = 10^7, 10^8

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("            Task 3: Max/Min Value and Index Benchmarks                   \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    long n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Running Benchmarks for N = %ld (%.2f MB array) ---\n",
           n, (double)(n * sizeof(int)) / (1024 * 1024));
    printf("=========================================================================\n");

    int *a = (int *)malloc(n * sizeof(int));

    if (!a)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", n);
      return 1;
    }

    // Initialize array with pattern and inject distinct max/min at specific locations
    for (long i = 0; i < n; i++)
    {
      a[i] = (int)(i % 5000);
    }

    long target_max_idx = n / 3;
    long target_min_idx = 2 * n / 3;
    a[target_max_idx] = 999999;
    a[target_min_idx] = -999999;

    // Serial Baseline
    int s_max_v, s_min_v;
    long s_max_i, s_min_i;
    double start_s = omp_get_wtime();
    find_max_min_serial(a, n, &s_max_v, &s_max_i, &s_min_v, &s_min_i);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Time: %8.6f s | Max: %d at idx %ld | Min: %d at idx %ld\n\n",
           time_s, s_max_v, s_max_i, s_min_v, s_min_i);

    // 1. Manual Per-Thread Reduction Method (Fast & Correct)
    printf("--- Method 1: Manual Per-Thread Reduction (Val + Index) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
    printf("----------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      int p_max_v, p_min_v;
      long p_max_i, p_min_i;

      double start_p = omp_get_wtime();
      find_max_min_idx_manual(a, n, &p_max_v, &p_max_i, &p_min_v, &p_min_i, threads);
      double time_p = omp_get_wtime() - start_p;

      int correct = (p_max_v == s_max_v && p_max_i == s_max_i &&
                     p_min_v == s_min_v && p_min_i == s_min_i);

      printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
             threads, time_p, time_s / time_p, correct ? "PASSED" : "FAILED");
    }
    printf("\n");

    // 2. OpenMP Value-only Reduction Method (max/min values only)
    printf("--- Method 2: Standard OpenMP reduction(max/min:val) (Values Only) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
    printf("----------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      int p_max_v, p_min_v;

      double start_p = omp_get_wtime();
      find_max_min_reduction(a, n, &p_max_v, &p_min_v, threads);
      double time_p = omp_get_wtime() - start_p;

      int correct = (p_max_v == s_max_v && p_min_v == s_min_v);

      printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
             threads, time_p, time_s / time_p, correct ? "PASSED" : "FAILED");
    }
    printf("\n");

    // 3. Critical-Section Method (Only for N = 10^7 due to slowness)
    if (n <= 10000000L)
    {
      printf("--- Method 3: #pragma omp critical (For Comparison) ---\n");
      printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
      printf("----------------------------------------------------\n");

      for (int t = 0; t < num_threads_list; t++)
      {
        int threads = thread_counts[t];
        int c_max_v;
        long c_max_i;

        double start_c = omp_get_wtime();
        find_max_idx_critical(a, n, &c_max_v, &c_max_i, threads);
        double time_c = omp_get_wtime() - start_c;

        int correct = (c_max_v == s_max_v && c_max_i == s_max_i);

        printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
               threads, time_c, time_s / time_c, correct ? "PASSED" : "FAILED");
      }
      printf("\n");
    }

    free(a);
  }

  return 0;
}