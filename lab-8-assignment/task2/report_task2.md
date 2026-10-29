# Task 2: Parallel Array Sorting (Odd-Even Transposition Sort) — Report

## 1. Algorithm & Synchronization Mechanics

- **Algorithm Selected:** Parallel Odd-Even Transposition Sort (Parallel Bubble Sort variant).
- **Phase Decomposition:**
  - **Even Phase:** Compares and swaps adjacent element pairs at even indices: `(0,1), (2,3), (4,5), ...`
  - **Odd Phase:** Compares and swaps adjacent element pairs at odd indices: `(1,2), (3,4), (5,6), ...`
- **Synchronization Requirements:**
  - Within any single phase (even or odd), all element pairs are completely independent and disjoint. Thus, elements can be compared and swapped concurrently across worker threads without data races.
  - An **implicit barrier synchronization** occurs at the end of each `#pragma omp for` loop. This guarantees that all threads complete the current phase before any thread begins the subsequent phase, maintaining algorithm correctness.

## 2. Benchmark Results

### Execution Time Table (seconds)

| Array Size ($N$) | Data Size (KB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5,000** | 19.53 KB | 0.018000s | 0.022000s | 0.118000s | 0.204000s | 0.437000s |
| **10,000** | 39.06 KB | 0.075000s | 0.075000s | 0.223000s | 0.482000s | 0.840000s |
| **20,000** | 78.12 KB | 0.296000s | 0.287000s | 0.481000s | 0.772000s | 1.752000s |
| **40,000** | 156.25 KB | 1.671000s | 1.416000s | 1.098000s | 1.183000s | 2.609000s |

### Speedup and Efficiency Table for Largest Array ($N = 40,000$)

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 1.416000s | 1.18x | 118.0% | PASSED |
| **2** | 1.098000s | 1.52x | 76.1% | PASSED |
| **4** | 1.183000s | 1.41x | 35.3% | PASSED |
| **8** | 2.609000s | 0.64x | 8.0% | PASSED |

## 3. Analysis & Questions

### Why is standard Bubble Sort difficult to parallelize, and how does Odd-Even Transposition Sort solve it?

- Standard Bubble Sort relies on an inner loop where each element comparison depends on the previous index update, creating immediate loop-carried data dependencies. Parallelizing standard Bubble Sort directly leads to severe data races.
- Odd-Even Transposition Sort breaks the algorithm into alternating Even and Odd phases. Because element pairs in any given phase do not overlap, comparisons and swaps within a phase are completely independent and can run in parallel without conflicts.

### What synchronization overhead exists in this parallel sorting algorithm?

- Each phase loop requires thread synchronization via the implicit OpenMP barrier at loop exit. For an array of size $N$, $N$ total phases occur, resulting in $N$ barrier synchronization events. While this ensures correctness, thread creation overhead is minimized by enclosing the entire phase loop inside a single persistent `#pragma omp parallel` region.