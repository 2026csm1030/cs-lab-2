# Task 3: Maximum and Minimum of an Array — Report

## 1. System Configuration
- **Compiler Flags:** `gcc -fopenmp -O2 task3_maxidx.c -o task3_runner`
- **Thread Management:** OpenMP `num_threads` directive clause / `OMP_NUM_THREADS`.

## 2. Benchmark Results

### Method 1: Manual Per-Thread Reduction (Finding Max/Min Values & Indexes)

#### Execution Time Table (seconds)
| Array Size ($N$) | Total Memory (MB) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^7$** | 38.15 MB | 0.014000s | 0.014000s | 0.010000s | 0.006000s | 0.005000s |
| **$10^8$** | 381.47 MB | 0.121000s | 0.119000s | 0.062000s | 0.046000s | 0.034000s |

#### Speedup Table for $N = 10^8$ ($S = T_{\text{serial}} / T_{\text{parallel}}$)
| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.119000s | 1.02x | 102.0% | PASSED |
| **2** | 0.062000s | 1.95x | 97.5% | PASSED |
| **4** | 0.046000s | 2.63x | 65.8% | PASSED |
| **8** | 0.034000s | 3.56x | 44.5% | PASSED |

### Method 2: Standard OpenMP `reduction(max/min: val)` (Values Only)

| Array Size ($N$) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^7$** | 0.014000s | 0.009000s | 0.006000s | 0.004000s | 0.004000s | 3.50x |
| **$10^8$** | 0.121000s | 0.083000s | 0.046000s | 0.029000s | 0.019000s | 6.37x |

### Method 3: Critical Section Approach vs. Manual Reduction ($N = 10^7$)

| Thread Count ($T$) | Manual Per-Thread Time (s) | Critical Section Time (s) | Slowdown Factor ($T_{\text{critical}} / T_{\text{manual}}$) |
| :---: | :---: | :---: | :---: |
| **1** | 0.014000s | 0.078000s | 5.57x slower |
| **2** | 0.010000s | 0.191000s | 19.10x slower |
| **4** | 0.006000s | 0.242000s | 40.33x slower |
| **8** | 0.005000s | 0.286000s | 57.20x slower |


## 3. Analysis & Questions

### Q1: Why can't you directly write `reduction(max: best_idx)` to get the index of the maximum?

- **Explanation:** The OpenMP `reduction(max: var)` construct evaluates the maximum among numerical scalar values. If applied to `best_idx` directly (`reduction(max: best_idx)`), OpenMP will compute the maximum index value (which is simply $N-1$ for the last loop iteration) rather than the index where `a[i]` achieves its maximum value. OpenMP C/C++ primitives do not natively support pair-wise reductions (e.g., `(value, index)` key-value reduction) without user-defined custom reductions (available in OpenMP 4.0+).

### Q2: Compare the performance of the `critical`-section approach vs. the manual per-thread reduction approach for finding `(max_val, max_idx)`. Which is faster, and why?

- **Comparison:** The **manual per-thread reduction approach is dramatically faster** than the critical section approach.
- **Why Critical Section is Slow:** Inside `find_max_idx_critical`, placing `#pragma omp critical` inside the loop body forces all threads to enter a synchronized lock on every iteration. This completely serializes loop execution, causes heavy cache-line bouncing on shared variables (`best_val` and `best_idx`), and introduces massive thread locking overhead.
- **Why Manual Reduction is Fast:** In the manual approach, each thread works on its allocated array chunk in isolation, tracking its own local `my_max_val` and `my_max_idx` inside fast CPU registers/L1 cache without any locks. At loop completion, a single short serial loop ($O(\text{num\_threads})$ iterations) combines the per-thread maximums.