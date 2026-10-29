#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define CUTOFF 2048

// Helper function to merge two sorted subarrays a[left..mid] and a[mid+1..right]

void merge(int *a, int *tmp, int left, int mid, int right)
{
  int i = left;
  int j = mid + 1;
  int k = left;

  while (i <= mid && j <= right)
  {
    if (a[i] <= a[j])
    {
      tmp[k++] = a[i++];
    }
    else
    {
      tmp[k++] = a[j++];
    }
  }

  while (i <= mid)
  {
    tmp[k++] = a[i++];
  }

  while (j <= right)
  {
    tmp[k++] = a[j++];
  }

  for (i = left; i <= right; i++)
  {
    a[i] = tmp[i];
  }
}

// 1. Serial Recursive Merge Sort

void merge_sort_serial(int *a, int *tmp, int left, int right)
{
  if (left >= right)
  {
    return;
  }

  int mid = left + (right - left) / 2;

  merge_sort_serial(a, tmp, left, mid);
  merge_sort_serial(a, tmp, mid + 1, right);
  merge(a, tmp, left, mid, right);
}

// 2. Parallel Merge Sort using OpenMP Tasks with Cutoff Threshold
void merge_sort_parallel(int *a, int *tmp, int left, int right)
{
  if (left >= right)
  {
    return;
  }

  // Cutoff threshold: Use serial sort for small subarrays to avoid task overhead
  if (right - left < CUTOFF)
  {
    merge_sort_serial(a, tmp, left, right);
    return;
  }

  int mid = left + (right - left) / 2;

#pragma omp task shared(a, tmp) firstprivate(left, mid)
  merge_sort_parallel(a, tmp, left, mid);

#pragma omp task shared(a, tmp) firstprivate(mid, right)
  merge_sort_parallel(a, tmp, mid + 1, right);

#pragma omp taskwait
  merge(a, tmp, left, mid, right);
}

// Wrapper for parallel merge sort entry point
void run_merge_sort_parallel(int *a, int *tmp, int n, int num_threads)
{
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single
    {
      merge_sort_parallel(a, tmp, 0, n - 1);
    }
  }
}

// Correctness Verification Helper
int verify_results(const int *a_serial, const int *a_parallel, int n)
{
  for (int i = 0; i < n; i++)
  {
    if (a_serial[i] != a_parallel[i])
    {
      printf("Mismatch at index %d: Serial = %d, Parallel = %d\n", i, a_serial[i], a_parallel[i]);
      return 0;
    }
  }

  for (int i = 0; i < n - 1; i++)
  {
    if (a_parallel[i] > a_parallel[i + 1])
    {
      printf("Array is not sorted at index %d: %d > %d\n", i, a_parallel[i], a_parallel[i + 1]);
      return 0;
    }
  }
  return 1;
}

int main(int argc, char *argv[])
{
  int sizes[] = {100000, 1000000, 10000000}; // N = 10^5, 10^6, 10^7

  int thread_counts[] = {1, 2, 4, 8};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  int num_threads_list = sizeof(thread_counts) / sizeof(thread_counts[0]);

  printf("=========================================================================\n");
  printf("         Task 1: Parallel Merge Sort using OpenMP Tasks                  \n");
  printf("=========================================================================\n\n");

  for (int s = 0; s < num_sizes; s++)
  {
    int n = sizes[s];

    printf("=========================================================================\n");
    printf("--- Running Benchmarks for N = %d (%.2f MB array) ---\n",
           n, (double)(n * sizeof(int)) / (1024 * 1024));
    printf("=========================================================================\n");

    int *a_orig = (int *)malloc(n * sizeof(int));
    int *a_serial = (int *)malloc(n * sizeof(int));
    int *a_parallel = (int *)malloc(n * sizeof(int));
    int *tmp = (int *)malloc(n * sizeof(int));

    if (!a_orig || !a_serial || !a_parallel || !tmp)
    {
      fprintf(stderr, "Memory allocation failed for N = %d\n", n);
      return 1;
    }

    // Initialize array with pseudo-random numbers
    srand(42);

    for (int i = 0; i < n; i++)
    {
      a_orig[i] = rand() % 1000000;
    }

    // Serial Baseline
    memcpy(a_serial, a_orig, n * sizeof(int));
    double start_s = omp_get_wtime();
    merge_sort_serial(a_serial, tmp, 0, n - 1);
    double time_s = omp_get_wtime() - start_s;

    printf("Serial Merge Sort Time: %8.6f seconds\n\n", time_s);
    printf("%-8s | %-12s | %-10s | %-12s | %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Verification");
    printf("-----------------------------------------------------------------------\n");

    for (int t = 0; t < num_threads_list; t++)
    {
      int threads = thread_counts[t];
      memcpy(a_parallel, a_orig, n * sizeof(int));

      double start_p = omp_get_wtime();
      run_merge_sort_parallel(a_parallel, tmp, n, threads);
      double time_p = omp_get_wtime() - start_p;

      int ok = verify_results(a_serial, a_parallel, n);
      double speedup = time_s / time_p;
      double eff = (speedup / threads) * 100.0;

      printf("%-8d | %-12.6f | %-10.2fx | %-11.1f%% | %-12s\n",
             threads, time_p, speedup, eff, ok ? "PASSED" : "FAILED");
    }

    free(a_orig);
    free(a_serial);
    free(a_parallel);
    free(tmp);
    printf("\n");
  }

  return 0;
}