# Task 3: Practice Multiple Reductions in OpenMP — Report

## 1. System Configuration

- **Problem Statement:** Count even and odd integers across array $a[N]$ using multiple OpenMP reduction variables (`even_count`, `odd_count`).
- **Compiler Flags:** `gcc -fopenmp -O2 task3_even_odd.c -o task3_runner`.
- **Verification:** Verified `even_count + odd_count == N` invariant and exact count match against serial execution for all problem sizes.

## 2. Benchmark Results

### Execution Time Table (seconds) for Method 1: `reduction(+:even, odd)`

| Array Size ($N$) | Memory (MB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | 0.38 MB | 0.000000s | 0.001000s | 0.001000s | 0.001000s | 0.001000s |
| **$10^6$** | 3.81 MB | 0.005000s | 0.005000s | 0.002000s | 0.001000s | 0.002000s |
| **$10^7$** | 38.15 MB | 0.048000s | 0.049000s | 0.024000s | 0.013000s | 0.009000s |

#### Speedup & Parallel Efficiency Table for $N = 10^7$ (`reduction`)

> **Serial Baseline Time:** $0.048000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Parallel Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.049000s | 0.98x | 98.0% | PASSED |
| **2** | 0.024000s | 2.00x | 100.0% | PASSED |
| **4** | 0.013000s | 3.69x | 92.3% | PASSED |
| **8** | 0.009000s | **5.33x** | 66.7% | PASSED |

### Method 1 (`reduction`) vs. Method 2 (`#pragma omp atomic`) Comparison ($N = 10^6$)

| Thread Count ($T$) | Reduction Time (s) | Atomic Time (s) | Atomic Speedup vs Serial | Slowdown Factor ($T_{\text{atomic}} / T_{\text{reduction}}$) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.005000s | 0.005000s | 1.00x | 1.00x |
| **2** | 0.002000s | 0.019000s | 0.26x | **9.50x slower** |
| **4** | 0.001000s | 0.024000s | 0.21x | **24.00x slower** |
| **8** | 0.002000s | 0.025000s | 0.20x | **12.50x slower** |

## 3. Analysis & Questions

### Q1: How can multiple variables be included in a reduction clause?
- Multiple reduction variables are specified within a single `reduction` clause by listing them separated by commas after the reduction operator, e.g., `#pragma omp parallel for reduction(+:even, odd)`.
- During loop execution, OpenMP automatically allocates private local accumulators for each variable (`even` and `odd`) for every worker thread. When the loop completes, OpenMP combines all local accumulators into the shared variables.

### Q2: Why is the reduction approach generally faster than using atomic updates?
- **`reduction` clause:** Each thread accumulates counts into thread-local variables allocated in fast CPU registers or L1 cache without any locks or inter-thread synchronization during loop iterations. Thread communication occurs only once at loop completion.
- **`#pragma omp atomic`:** Forces atomic read-modify-write hardware instructions across all threads on every single loop iteration. At 8 threads ($N = 10^6$), atomic operations cause heavy cache-line bouncing on shared memory addresses (`even` and `odd`), degrading performance to **0.20x speedup** ($0.025\text{s}$ vs $0.002\text{s}$ for reduction, making atomic **12.5x slower**).

### Q3: Why must the sum of the even and odd counts equal $N$?
- Every element $a[i]$ in an integer array is strictly either even ($a[i] \pmod 2 = 0$) or odd ($a[i] \pmod 2 \neq 0$). Since these two conditions are mutually exclusive and collectively exhaustive across the entire array iteration space $[0, N-1]$, every element increments exactly one counter, ensuring that $\text{even\_count} + \text{odd\_count} = N$.

### Q4: Does increasing the number of threads always improve performance?
- No. Array counting is fundamentally a **memory-bandwidth bound** operation with low operational intensity (1 modulo operation and 1 addition per 4-byte element).
- At smaller problem sizes ($N = 10^6$), performance peaks at 4 threads ($0.001\text{s}$, 5.00x speedup) before degrading slightly at 8 threads ($0.002\text{s}$) due to memory bus saturation and OpenMP thread management overhead. For large $N = 10^7$, scaling continues up to 8 threads ($0.009\text{s}$, 5.33x speedup) as computation time dominates overhead.