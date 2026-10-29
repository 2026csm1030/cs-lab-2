#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 1. Serial Sum
long sum_serial(const long *a, long n)
{
  long sum = 0;

  for (long i = 0; i < n; i++)
  {
    sum += a[i];
  }

  return sum;
}

// 2. Naive Parallel Sum (BROKEN due to data race)
long sum_naive_parallel_broken(const long *a, long n, int num_threads)
{
  long sum = 0;

#pragma omp parallel for num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    sum += a[i]; // Unsynchronized concurrent write: Data Race!
  }

  return sum;
}

// 3. Reduction-based Parallel Sum (Correct & High Performance)
long sum_reduction(const long *a, long n, int num_threads)
{
  long sum = 0;

#pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    sum += a[i];
  }

  return sum;
}

// 4. Critical Section Parallel Sum (Correct but Slow)
long sum_critical(const long *a, long n, int num_threads)
{
  long sum = 0;

#pragma omp parallel for num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
#pragma omp critical
    {
      sum += a[i];
    }
  }

  return sum;
}

int main(int argc, char *argv[])
{
  long sizes[] = {10000000L, 100000000L}; // N = 10^7, 10^8

  int thread_counts[] = {1, 2, 4, 8};

  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("               Task 2: Sum of Array Elements Benchmarks                  \n");
  printf("=========================================================================\n\n");

  // Demonstration of Naive Data Race Behavior
  printf("--- Part 1: Demonstrating Unsafe Naive Parallel Sum (N = 10^7, Threads = 4) ---\n");

  long demo_n = 10000000L;
  long *demo_a = (long *)malloc(demo_n * sizeof(long));

  for (long i = 0; i < demo_n; i++)
  {
    demo_a[i] = 1; // Expected total sum = 10,000,000
  }

  long expected_sum = sum_serial(demo_a, demo_n);
  printf("Expected (Serial) Sum : %ld\n", expected_sum);

  for (int run = 1; run <= 5; run++)
  {
    long naive_res = sum_naive_parallel_broken(demo_a, demo_n, 4);

    printf("Naive Run #%d           : %ld  (%s)\n",
           run, naive_res, (naive_res == expected_sum) ? "CORRECT" : "WRONG / DATA RACE");
  }

  free(demo_a);
  printf("\n");

  // Benchmark across sizes and thread counts
  for (int s = 0; s < num_sizes; s++)
  {
    long n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Running Benchmarks for N = %ld (%.2f MB array) ---\n",
           n, (double)(n * sizeof(long)) / (1024 * 1024));
    printf("=========================================================================\n");

    long *a = (long *)malloc(n * sizeof(long));

    if (!a)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", n);
      return 1;
    }

    for (long i = 0; i < n; i++)
    {
      a[i] = (i % 10) + 1; // Arbitrary small integers
    }

    // Measure Serial Baseline
    double start_serial = omp_get_wtime();
    long target_sum = sum_serial(a, n);
    double time_serial = omp_get_wtime() - start_serial;

    printf("Serial Execution Time: %8.6f seconds | Total Sum: %ld\n\n", time_serial, target_sum);

    // Benchmark OpenMP Reduction
    printf("--- Method: OpenMP reduction(+:sum) ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-10s\n", "Threads", "Time (sec)", "Speedup", "Verification", "Calculated Sum");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_red = omp_get_wtime();
      long red_sum = sum_reduction(a, n, threads);
      double time_red = omp_get_wtime() - start_red;

      double speedup = time_serial / time_red;
      int passed = (red_sum == target_sum);

      printf("%-8d | %-12.6f | %-10.2fx | %-12s | %ld\n",
             threads, time_red, speedup, passed ? "PASSED" : "FAILED", red_sum);
    }
    printf("\n");

    // Benchmark Critical Section (only for N = 10^7 to avoid excessive wait time)
    if (n <= 10000000L)
    {
      printf("--- Method: #pragma omp critical (For Comparison) ---\n");
      printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
      printf("----------------------------------------------------\n");

      for (int t = 0; t < num_threads_list; t++)
      {
        int threads = thread_counts[t];

        double start_crit = omp_get_wtime();
        long crit_sum = sum_critical(a, n, threads);
        double time_crit = omp_get_wtime() - start_crit;

        double speedup = time_serial / time_crit;
        int passed = (crit_sum == target_sum);

        printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
               threads, time_crit, speedup, passed ? "PASSED" : "FAILED");
      }
      printf("\n");
    }

    free(a);
  }

  return 0;
}