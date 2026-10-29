# Task 4: Matrix Addition — Report

## 1. System Configuration & Data Layout
- **Memory Representation:** Flattened row-major 1D arrays of size $N \times N$ (`double`), indexed as `[i * N + j]`, providing contiguous memory layout and high cache locality.
- **Compiler Flags:** `gcc -fopenmp -O2 task4_matadd.c -o task4_runner`
- **Thread Control:** OpenMP `num_threads` clause.

## 2. Benchmark Results

### Small Matrix ($N = 8 \times 8 = 64$ elements)

#### Execution Time & Speedup Comparison
| Thread Count ($T$) | Outer Loop Time (s) | Outer Speedup | Collapse(2) Time (s) | Collapse(2) Speedup | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.000000s | N/A ($0.00\text{s}$ baseline) | 0.000000s | N/A | PASSED |
| **2** | 0.001000s | 0.00x | 0.000000s | N/A | PASSED |
| **4** | 0.000000s | N/A | 0.000000s | N/A | PASSED |
| **8** | 0.000000s | N/A | 0.001000s | 0.00x | PASSED |


### Large Matrix ($N = 4096 \times 4096 \approx 16.78\text{ M}$ elements)

#### Execution Time & Speedup Comparison
| Thread Count ($T$) | Outer Loop Time (s) | Outer Speedup | Collapse(2) Time (s) | Collapse(2) Speedup | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.059000s | 0.90x | 0.041000s | 1.29x | PASSED |
| **2** | 0.040000s | 1.32x | 0.042000s | 1.26x | PASSED |
| **4** | 0.040000s | 1.33x | 0.055000s | 0.96x | PASSED |
| **8** | 0.038000s | 1.39x | 0.040000s | 1.33x | PASSED |

## 3. Analysis & Questions

### Q1: For the small-$N$ case ($N = 8$), does `collapse(2)` help compared to parallelizing only the outer loop? Explain why in terms of iteration space.

- **Iteration Distribution Analysis:**
  - **Outer Loop Only (`#pragma omp parallel for`):** Only $N = 8$ total iterations are available for thread distribution. When running with 8 threads, each thread gets exactly 1 outer iteration (processing 8 elements). When running with thread counts $> 8$, remaining threads remain idle because the loop bound limits parallelism to 8 units.
  - **Collapsed Loops (`#pragma omp parallel for collapse(2)`):** The runtime flattens the two nested loops into a single virtual iteration space of $N \times N = 8 \times 8 = 64$ independent iterations. This provides much finer grain workload units to distribute among worker threads.
- **Performance Impact:** Although `collapse(2)` creates 64 iterations, for extremely small $N = 8$, thread creation/dispatch overhead often dominates overall computation time. However, structurally `collapse(2)` prevents thread starvation when $N < \text{num\_threads}$.

### Q2: For the large-$N$ case ($N = 4096$), does `collapse(2)` still make a measurable difference, or is parallelizing the outer loop alone already enough to keep all threads busy?

- **Analysis:** For $N = 4096$, parallelizing the outer loop alone creates 4,096 coarse tasks. Distributing 4,096 row-level tasks among 2, 4, or 8 threads gives thousands of iterations per thread, providing ample workload balance to keep all CPU cores 100% utilized.
- **Result:** Collapsing to $4096 \times 4096 \approx 1.67 \times 10^7$ iterations provides no significant performance improvement over outer-loop parallelization because memory bandwidth limits (RAM throughput) become the bottleneck long before iteration starvation occurs.