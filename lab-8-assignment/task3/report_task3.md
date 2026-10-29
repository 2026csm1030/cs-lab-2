# Task 3: Histogram Calculation using OpenMP — Report

## 1. System Configuration

- **Bin Range:** Integers from 0 to 9 (`NUM_BINS = 10`).
- **Compiler Flags:** `gcc -fopenmp -O2 task3_histogram.c -o task3_runner`.
- **Verification:** Computed frequency counts were verified against serial execution for exact match across all 10 bins.

## 2. Benchmark Results

### Method 1: Thread-Local Private Buckets + Final Critical Pass

#### Execution Time Table (seconds)
| Array Size ($N$) | Total Memory (MB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^7$** | 38.15 MB | 0.007000s | 0.008000s | 0.005000s | 0.003000s | 0.003000s |
| **$10^8$** | 381.47 MB | 0.066000s | 0.074000s | 0.036000s | 0.021000s | 0.015000s |

#### Speedup & Efficiency Table ($N = 10^8$)
> Baseline Serial Time: $0.066000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.074000s | 0.89x | 89.2% | PASSED |
| **2** | 0.036000s | 1.83x | 91.7% | PASSED |
| **4** | 0.021000s | 3.14x | 78.6% | PASSED |
| **8** | 0.015000s | 4.40x | 55.0% | PASSED |

### Method 2: OpenMP Array Reduction Clause (`reduction(+:hist[:10])`)

| Array Size ($N$) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^7$** | 0.007000s | 0.007000s | 0.005000s | 0.003000s | 0.004000s | 1.75x |
| **$10^8$** | 0.066000s | 0.061000s | 0.032000s | 0.018000s | 0.015000s | **4.40x** |

### Method 3: Critical Section Per Element vs. Thread-Local Method ($N = 10^7$)

| Thread Count ($T$) | Thread-Local Time (s) | Critical Section Time (s) | Slowdown Factor ($T_{\text{critical}} / T_{\text{local}}$) |
| :---: | :---: | :---: | :---: |
| **1** | 0.008000s | 0.086000s | 10.75x slower |
| **2** | 0.005000s | 0.148000s | 29.60x slower |
| **4** | 0.003000s | 0.202000s | 67.33x slower |
| **8** | 0.003000s | 0.312000s | **104.00x slower** |

## 3. Analysis & Discussion

### How are data races prevented in the histogram calculation?

- **Data Race Hazard:** In a naive implementation (`#pragma omp parallel for`), multiple threads read `a[i]` and attempt to update the shared bin `hist[a[i]]++` simultaneously. Because 10 bin locations are shared among all threads, updates collide frequently, causing lost increments.
- **Thread-Local Accumulation:** To prevent race conditions efficiently, each thread creates a private array `local_hist[10]` initialized to zero. Threads accumulate frequencies independently in their private buffers without locks or cross-thread interference.
- **Final Reduction Step:** After loop completion, each thread enters a brief critical section (`#pragma omp critical`) to add its 10 thread-local bin counts into the global `hist` array. Since critical section access happens only once per thread rather than once per element, lock contention overhead is negligible.