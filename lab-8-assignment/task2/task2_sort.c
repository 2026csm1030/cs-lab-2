#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Helper swap function
static inline void swap(int *x, int *y)
{
  int temp = *x;
  *x = *y;
  *y = temp;
}

// 1. Serial Bubble / Odd-Even Transposition Sort
void sort_serial(int *a, int n)
{
  for (int phase = 0; phase < n; phase++)
  {
    if (phase % 2 == 0)
    { // Even Phase
      for (int i = 0; i < n - 1; i += 2)
      {
        if (a[i] > a[i + 1])
        {
          swap(&a[i], &a[i + 1]);
        }
      }
    }
    else
    { // Odd Phase
      for (int i = 1; i < n - 1; i += 2)
      {
        if (a[i] > a[i + 1])
        {
          swap(&a[i], &a[i + 1]);
        }
      }
    }
  }
}

// 2. Parallel Odd-Even Transposition Sort using OpenMP
void sort_parallel(int *a, int n, int num_threads)
{
#pragma omp parallel num_threads(num_threads)
  {
    for (int phase = 0; phase < n; phase++)
    {
      if (phase % 2 == 0)
      {
        // Even Phase: compare (0,1), (2,3), ...
#pragma omp for schedule(static)

        for (int i = 0; i < n - 1; i += 2)
        {
          if (a[i] > a[i + 1])
          {
            swap(&a[i], &a[i + 1]);
          }
        }
      }
      else
      {
        // Odd Phase: compare (1,2), (3,4), ...
#pragma omp for schedule(static)

        for (int i = 1; i < n - 1; i += 2)
        {
          if (a[i] > a[i + 1])
          {
            swap(&a[i], &a[i + 1]);
          }
        }
      }
      // Implicit barrier at the end of '#pragma omp for' ensures
      // all threads finish the current phase before the next phase begins.
    }
  }
}

int verify_sorted(const int *a, const int *ref, int n)
{
  for (int i = 0; i < n; i++)
  {
    if (a[i] != ref[i])
    {
      return 0; // Mismatch found
    }
  }
  for (int i = 0; i < n - 1; i++)
  {
    if (a[i] > a[i + 1])
    {
      return 0; // Not sorted
    }
  }
  return 1; // Passed
}

int main(int argc, char *argv[])
{
  // Array sizes for O(N^2) sorting benchmarks
  int sizes[] = {5000, 10000, 20000, 40000};

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("            Task 2: Parallel Odd-Even Transposition Sort                 \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    int n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Sorting Array of Size N = %d (%.2f KB) ---\n",
           n, (double)(n * sizeof(int)) / 1024);
    printf("=========================================================================\n");

    int *a_orig = (int *)malloc(n * sizeof(int));
    int *a_serial = (int *)malloc(n * sizeof(int));
    int *a_parallel = (int *)malloc(n * sizeof(int));

    if (!a_orig || !a_serial || !a_parallel)
    {
      fprintf(stderr, "Memory allocation failed for N = %d\n", n);
      return 1;
    }

    // Initialize array with pseudo-random numbers
    srand(42);
    for (int i = 0; i < n; i++)
    {
      a_orig[i] = rand() % 100000;
    }

    // Serial Execution Baseline
    memcpy(a_serial, a_orig, n * sizeof(int));
    double start_s = omp_get_wtime();
    sort_serial(a_serial, n);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Sort Time: %8.6f seconds\n\n", time_s);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    // Parallel Execution across thread counts
    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      memcpy(a_parallel, a_orig, n * sizeof(int));

      double start_p = omp_get_wtime();
      sort_parallel(a_parallel, n, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_sorted(a_parallel, a_serial, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }

    free(a_orig);
    free(a_serial);
    free(a_parallel);
    printf("\n");
  }

  return 0;
}