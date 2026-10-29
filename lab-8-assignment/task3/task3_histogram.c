#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_BINS 10

// 1. Serial Histogram Calculation
void histogram_serial(const int *a, long n, long *hist)
{
  for (int b = 0; b < NUM_BINS; b++)
  {
    hist[b] = 0;
  }
  for (long i = 0; i < n; i++)
  {
    hist[a[i]]++;
  }
}

// 2. Parallel Histogram with Critical Section (Correct but Slow)
void histogram_critical(const int *a, long n, long *hist, int num_threads)
{
  for (int b = 0; b < NUM_BINS; b++)
  {
    hist[b] = 0;
  }

#pragma omp parallel for num_threads(num_threads)

  for (long i = 0; i < n; i++)
  {
    int val = a[i];

#pragma omp critical
    {
      hist[val]++;
    }
  }
}

// 3. Parallel Histogram with Thread-Local Private Buckets (Fast & Race-Free)
void histogram_local_private(const int *a, long n, long *hist, int num_threads)
{
  for (int b = 0; b < NUM_BINS; b++)
  {
    hist[b] = 0;
  }

#pragma omp parallel num_threads(num_threads)
  {
    long local_hist[NUM_BINS] = {0}; // Thread-private local histogram accumulator

#pragma omp for schedule(static)

    for (long i = 0; i < n; i++)
    {
      local_hist[a[i]]++;
    }

// Combine local histograms into global histogram
#pragma omp critical
    {
      for (int b = 0; b < NUM_BINS; b++)
      {
        hist[b] += local_hist[b];
      }
    }
  }
}

// 4. Parallel Histogram with OpenMP Array Reduction (OpenMP 4.5+)
void histogram_reduction(const int *a, long n, long *hist, int num_threads)
{
  for (int b = 0; b < NUM_BINS; b++)
  {
    hist[b] = 0;
  }

#pragma omp parallel for reduction(+ : hist[ : NUM_BINS]) num_threads(num_threads)
  for (long i = 0; i < n; i++)
  {
    hist[a[i]]++;
  }
}

int verify_histogram(const long *ref, const long *test)
{
  for (int b = 0; b < NUM_BINS; b++)
  {
    if (ref[b] != test[b])
    {
      return 0; // Mismatch found
    }
  }
  return 1; // Verification Passed
}

int main(int argc, char *argv[])
{
  long sizes[] = {10000000L, 100000000L}; // N = 10^7, 10^8

  int thread_counts[] = {1, 2, 4, 8};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("                Task 3: Histogram Calculation Benchmarks                  \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    long n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Running Histogram for N = %ld (%.2f MB array) ---\n",
           n, (double)(n * sizeof(int)) / (1024 * 1024));
    printf("=========================================================================\n");

    int *a = (int *)malloc(n * sizeof(int));

    if (!a)
    {
      fprintf(stderr, "Memory allocation failed for N = %ld\n", n);
      return 1;
    }

    // Initialize array with random integers between 0 and 9
    for (long i = 0; i < n; i++)
    {
      a[i] = i % 10;
    }

    long hist_serial[NUM_BINS];
    long hist_test[NUM_BINS];

    // Serial Execution Baseline
    double start_s = omp_get_wtime();
    histogram_serial(a, n, hist_serial);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Execution Time: %8.6f seconds\n\n", time_s);

    // Method 1: Local Private Histogram + Critical Reduction
    printf("--- Method 1: Thread-Local Private Buckets ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      histogram_local_private(a, n, hist_test, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_histogram(hist_serial, hist_test);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 2: OpenMP 4.5 Array Reduction Clause
    printf("--- Method 2: OpenMP reduction(+:hist[:10]) Clause ---\n");
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];

      double start_p = omp_get_wtime();
      histogram_reduction(a, n, hist_test, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_histogram(hist_serial, hist_test);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }
    printf("\n");

    // Method 3: Critical Section Per Element (Demonstration for N = 10^7)
    if (n <= 10000000L)
    {
      printf("--- Method 3: #pragma omp critical inside inner loop (Slow) ---\n");
      printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Verification");
      printf("----------------------------------------------------\n");

      for (int t = 0; t < num_threads_list; t++)
      {
        int threads = thread_counts[t];

        double start_c = omp_get_wtime();
        histogram_critical(a, n, hist_test, threads);
        double time_c = omp_get_wtime() - start_c;

        int ok = verify_histogram(hist_serial, hist_test);
        double speedup = time_s / time_c;

        printf("%-8d | %-12.6f | %-10.2fx | %-12s\n",
               threads, time_c, speedup, ok ? "PASSED" : "FAILED");
      }
      printf("\n");
    }

    free(a);
  }

  return 0;
}