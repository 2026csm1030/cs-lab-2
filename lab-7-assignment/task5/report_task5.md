# Task 5: Matrix-Matrix Multiplication (Naive & Optimized) — Report

## 1. System Configuration & Data Layout

- **Memory Representation:** Flattened row-major 1D arrays of size $N \times N$ (`double`), indexed via `[i * N + j]`.
- **Compiler Flags:** `gcc -fopenmp -O2 task5_matmul.c -o task5_runner -lm`
- **Verification Threshold:** Floating-point tolerance check with $\vert{}C_{\text{ref}}[i] - C_{\text{test}}[i]\vert{} < 10^{-6}$.

## 2. Benchmark Results

### Method 1: Outermost Loop Parallelization (`#pragma omp parallel for` on $i$)

| Matrix Size ($N$) | Serial Baseline ($i\text{-}j\text{-}k$) | 1 Thread (s) | 2 Threads (s) | 4 Threads (s) | 8 Threads (s) | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | 0.063000s | 0.062000s | 0.034000s | 0.016000s | 0.016000s | 3.94x |
| **512** | 1.150000s | 1.133000s | 0.554000s | 0.282000s | 0.169000s | 6.80x |
| **1024** | 7.836000s | 6.920000s | 3.284000s | 1.550000s | 0.891000s | 8.79x |

#### Speedup & Efficiency Table for $N = 1024$

> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Parallel Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 6.920000s | 1.13x | 113.2% | PASSED |
| **2** | 3.284000s | 2.39x | 119.3% | PASSED |
| **4** | 1.550000s | 5.06x | 126.4% | PASSED |
| **8** | 0.891000s | 8.79x | 109.9% | PASSED |

---

### Method 2: Collapsed Loops (`#pragma omp parallel for collapse(2)` on $i, j$)

| Matrix Size ($N$) | Serial Baseline ($i\text{-}j\text{-}k$) | 1 Thread (s) | 2 Threads (s) | 4 Threads (s) | 8 Threads (s) | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | 0.063000s | 0.070000s | 0.030000s | 0.044000s | 0.020000s | 3.15x |
| **512** | 1.150000s | 1.157000s | 0.564000s | 0.274000s | 0.168000s | 6.85x |
| **1024** | 7.836000s | 6.988000s | 3.348000s | 1.568000s | 0.861000s | 9.10x |

---

### Bonus Method: Swapped Loop Order ($i, k, j$) Performance Comparison ($N = 1024$)

| Matrix Size ($N$) | $i\text{-}k\text{-}j$ Serial Time | 1 Thread (s) | 2 Threads (s) | 4 Threads (s) | 8 Threads (s) | Max Speedup vs Original Baseline ($i\text{-}j\text{-}k$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | 0.016000s | 0.017000s | 0.006000s | 0.004000s | 0.003000s | **21.00x** |
| **512** | 0.069000s | 0.070000s | 0.037000s | 0.029000s | 0.018000s | **63.89x** |
| **1024** | 0.404000s | 0.409000s | 0.239000s | 0.183000s | 0.132000s | **59.36x** |

---

## 3. Analysis & Questions

### Q1: Why is it safe to parallelize the $i$ loop directly (no race condition), even though the innermost $k$ loop is doing a summation into $C[i][j]$?

- **Explanation:** In matrix multiplication $C[i][j] = \sum_k A[i][k] \times B[k][j]$, every row index $i$ corresponds to a completely disjoint row slice of output matrix $C$. When the outer $i$ loop is parallelized across threads, each thread receives a distinct subset of row indices $i$. Therefore, no two worker threads ever attempt to read or write to the same element $C[i][j]$ simultaneously, eliminating any potential data race on $C$. Accumulating into a thread-local scalar accumulator variable `sum` inside the inner loop further eliminates unnecessary shared memory traffic.

### Q2: What happens to correctness if you try to parallelize the $k$ loop directly with a plain `parallel for` (no reduction)? Explain the race condition that would occur on $C[i][j]$.

- **Explanation:** The innermost loop index $k$ iterates through the reduction terms for a fixed output element $C[i][j]$. If the $k$ loop were parallelized directly with `#pragma omp parallel for`, multiple threads would concurrently attempt to update the exact same destination cell $C[i][j] += A[i][k] \times B[k][j]$. Because addition is not an atomic operation at the hardware assembly level, concurrent writes would create lost updates and severe data races, causing corrupted, non-deterministic matrix results.

### Q3: Compare `parallel for` on $i$ alone vs. `collapse(2)` on $(i, j)$. For which values of $N$ (relative to thread count) would you expect `collapse(2)` to matter most?

- **Analysis:**
  - Parallelizing the outer $i$ loop yields $N$ independent work chunks. If $N \ge \text{thread count}$ (e.g., $N = 256, 512, 1024$ on an 8-thread system), $N$ provides enough parallel work units to keep all threads fully utilized without loop collapsing.
  - `collapse(2)` combines $i$ and $j$ loops into $N^2$ independent tasks.
- **When `collapse(2)` Matters Most:** `collapse(2)` is crucial when $N$ is small relative to the thread count (e.g., $N = 4$ or $N = 8$ on a 64-thread processor). In that scenario, parallelizing only $i$ leaves 56 threads idle ($N < T$). Collapsing $i$ and $j$ yields $N^2 = 64$ iterations, allowing all 64 threads to execute concurrently.

### Q4 (Bonus): Why does swapping loop order to $i, k, j$ improve performance?

- **Explanation:** In the standard $i, j, k$ order, the inner loop accesses $B[k][j]$ column-wise across $k$, jumping across memory strides of size $N \times \text{sizeof(double)}$. This causes frequent CPU cache misses and poor spatial locality.
- In the $i, k, j$ order, the innermost loop iterates across $j$, accessing elements $B[k][j]$ and $C[i][j]$ sequentially along contiguous memory rows. This dramatically improves spatial cache locality, maximizes hardware cache prefetching, and allows AVX/SIMD auto-vectorization, resulting in substantial speedup for both serial baseline and parallel runs.