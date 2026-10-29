#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 1. Serial Implementation
void count_even_odd_serial(const int *a, long n, long *even_count, long *odd_count)
{
  long even = 0;
  long odd = 0;

  for (long i = 0; i < n; i++)
  {
    if (a[i] % 2 == 0)
    {
      even++;
    }
    else
    {
      odd++;
    }
  }

  *even_count = even;
  *odd_count = odd;
}

// 2. Parallel Implementation using Multiple Reductions: reduction(+:even, odd)
void count_even_odd_reduction(const int *a, long n, long *even_count, long *odd_count, int num_threads)
{
  long even = 0;
  long odd = 0;

#pragma omp parallel for reduction(+ : even, odd) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    if (a[i] % 2 == 0)
    {
      even++;
    }
    else
    {
      odd++;
    }
  }

  *even_count = even;
  *odd_count = odd;
}

// 3. Parallel Implementation using #pragma omp atomic
void count_even_odd_atomic(const int *a, long n, long *even_count, long *odd_count, int num_threads)
{
  long even = 0;
  long odd = 0;

#pragma omp parallel for num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    if (a[i] % 2 == 0)
    {
#pragma omp atomic
      even++;
    }
    else
    {
#pragma omp atomic
      odd++;
    }
  }

  *even_count = even;
  *odd_count = odd;
}

int verify_counts(long ref_even, long ref_odd, long test_even, long test_odd, long n)
{
  if (ref_even != test_even || ref_odd != test_odd)
  {
    printf("Mismatch! Ref: (Even=%ld, Odd=%ld), Test: (Even=%ld, Odd=%ld)\n",
           ref_even, ref_odd, test_even, test_odd);
    return 0;
  }
  if ((test_even + test_odd) != n)
  {
    printf("Sum invariant violated! even (%ld) + odd (%ld) != N (%ld)\n", test_even, test_odd, n);
    return 0;
  }
  return 1;
}

int main(int argc, char *argv[])
{
  long sizes[] = {100000L, 1000000L, 10000000L}; // N = 10^5, 10^6, 10^7

  int thread_counts[] = {1, 2, 4, 8};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("        Task 3: Multiple Reductions (Even/Odd Counting) Benchmarks       \n");
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

    // Initialize array with pseudo-random numbers
    srand(42);

    for (long i = 0; i < n; i++)
    {
      a[i] = rand();
    }

    // Serial Baseline
    long ref_even = 0, ref_odd = 0;
    double start_s = omp_get_wtime();
    count_even_odd_serial(a, n, &ref_even, &ref_odd);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Baseline Time: %8.6f s | Even: %ld, Odd: %ld\n\n", time_s, ref_even, ref_odd);

    // Method 1: Multiple Reductions
    printf("--- Method 1: reduction(+:even, odd) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      long test_even = 0, test_odd = 0;

      double start_p = omp_get_wtime();
      count_even_odd_reduction(a, n, &test_even, &test_odd, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_counts(ref_even, ref_odd, test_even, test_odd, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 2: #pragma omp atomic (Only for N <= 10^6 to avoid excessive wait times)
    if (n <= 1000000L)
    {
      printf("--- Method 2: #pragma omp atomic (For Comparison) ---\n");
      printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
      printf("-----------------------------------------------------------------------\n");

      for (int t = 0; t < num_threads_list; t++)
      {
        int threads = thread_counts[t];
        long test_even = 0, test_odd = 0;

        double start_p = omp_get_wtime();
        count_even_odd_atomic(a, n, &test_even, &test_odd, threads);
        double time_p = omp_get_wtime() - start_p;

        int ok = verify_counts(ref_even, ref_odd, test_even, test_odd, n);
        double speedup = time_s / time_p;
        double eff = (speedup / threads) * 100.0;

        printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
               threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
      }
      printf("\n");
    }

    free(a);
  }

  return 0;
}