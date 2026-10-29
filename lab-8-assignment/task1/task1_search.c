#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 1. Serial Linear Search
long search_serial(const int *a, long n, int target)
{
  for (long i = 0; i < n; i++)
  {
    if (a[i] == target)
    {
      return i; // Return first matching index
    }
  }
  return -1; // Target not found
}

// 2. OpenMP Parallel Search (Find First Occurrence with Early Exit Flag)
long search_parallel(const int *a, long n, int target, int num_threads)
{
  long found_idx = -1;

#pragma omp parallel num_threads(num_threads)
  {
#pragma omp for schedule(static)
    for (long i = 0; i < n; i++)
    {
      // Check flag to avoid unnecessary work if target already found by another thread
      if (found_idx != -1 && i > found_idx)
      {
        continue;
      }

      if (a[i] == target)
      {
#pragma omp critical
        {
          if (found_idx == -1 || i < found_idx)
          {
            found_idx = i; // Save the smallest matching index
          }
        }
      }
    }
  }

  return found_idx;
}

int main(int argc, char *argv[])
{
  long sizes[] = {100000L, 1000000L, 10000000L, 100000000L}; // N = 10^5, 10^6, 10^7, 10^8
  int thread_counts[] = {1, 2, 4, 8};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                  Task 1: Array Search Benchmarks                        \n");
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

    // Initialize array with arbitrary values
    for (long i = 0; i < n; i++)
    {
      a[i] = (int)(i % 1000) + 1;
    }

    // Scenario A: Worst Case (Target near the end of the array)
    long target_idx_worst = n - 1000;
    int target_val_worst = 999999;
    a[target_idx_worst] = target_val_worst; // Inject target

    printf("\n>>> Scenario A: Target Near End (Index %ld - Worst Case) <<<\n", target_idx_worst);

    // Serial Baseline
    double start_s = omp_get_wtime();
    long s_idx = search_serial(a, n, target_val_worst);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Search Time: %8.6f seconds | Found Index: %ld\n", time_s, s_idx);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Found Index", "Verification");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      long p_idx = search_parallel(a, n, target_val_worst, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = (p_idx == s_idx);
      double speedup = time_s / time_p;

      printf("%-8d | %-12.6f | %-10.2fx | %-12ld | %-12s\n",
             threads, time_p, speedup, p_idx, ok ? "PASSED" : "FAILED");
    }

    // Scenario B: Best/Average Case (Target in the middle of the array)
    long target_idx_mid = n / 2;
    int target_val_mid = 888888;
    a[target_idx_mid] = target_val_mid; // Inject target

    printf("\n>>> Scenario B: Target Near Middle (Index %ld - Average Case) <<<\n", target_idx_mid);

    start_s = omp_get_wtime();
    s_idx = search_serial(a, n, target_val_mid);
    time_s = omp_get_wtime() - start_s;

    printf("Serial Search Time: %8.6f seconds | Found Index: %ld\n", time_s, s_idx);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Found Index", "Verification");
    printf("--------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      long p_idx = search_parallel(a, n, target_val_mid, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = (p_idx == s_idx);
      double speedup = time_s / time_p;

      printf("%-8d | %-12.6f | %-10.2fx | %-12ld | %-12s\n",
             threads, time_p, speedup, p_idx, ok ? "PASSED" : "FAILED");
    }

    free(a);
    printf("\n");
  }

  return 0;
}