# Task 1: Array Search using OpenMP — Report

## 1. System Configuration

- **Compiler Flags:** `gcc -fopenmp -O2 task1_search.c -o task1_runner`
- **Thread Management:** OpenMP `num_threads` clause / runtime environment variables.
- **Verification:** Both serial and parallel implementations returned identical matching index positions across all array sizes.

## 2. Benchmark Results

### Scenario A: Worst Case (Target Near Array End)

#### Execution Time Table (seconds)
| Array Size ($N$) | Array Size (MB) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | 0.38 MB | 0.000000s | 0.000000s | 0.001000s | 0.000000s | 0.001000s |
| **$10^6$** | 3.81 MB | 0.001000s | 0.001000s | 0.001000s | 0.002000s | 0.003000s |
| **$10^7$** | 38.15 MB | 0.005000s | 0.009000s | 0.007000s | 0.004000s | 0.005000s |
| **$10^8$** | 381.47 MB | 0.049000s | 0.093000s | 0.052000s | 0.038000s | 0.027000s |

#### Speedup & Efficiency Table ($N = 10^8$, Target at Index $99,999,000$)
> **Serial Baseline ($T_{\text{serial}}$):** $0.049000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency | Found Index | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.093000s | 0.53x | 53.0% | 99999000 | PASSED |
| **2** | 0.052000s | 0.94x | 47.0% | 99999000 | PASSED |
| **4** | 0.038000s | 1.29x | 32.3% | 99999000 | PASSED |
| **8** | 0.027000s | 1.81x | 22.6% | 99999000 | PASSED |

---

### Scenario B: Average Case (Target Near Array Middle)

#### Execution Time Table (seconds)
| Array Size ($N$) | Array Size (MB) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | 0.38 MB | 0.000000s | 0.000000s | 0.001000s | 0.000000s | 0.001000s |
| **$10^6$** | 3.81 MB | 0.000000s | 0.001000s | 0.001000s | 0.001000s | 0.001000s |
| **$10^7$** | 38.15 MB | 0.004000s | 0.010000s | 0.005000s | 0.004000s | 0.003000s |
| **$10^8$** | 381.47 MB | 0.027000s | 0.091000s | 0.043000s | 0.030000s | 0.021000s |

#### Speedup & Efficiency Table ($N = 10^8$, Target at Index $50,000,000$)
> **Serial Baseline ($T_{\text{serial}}$):** $0.027000\text{ seconds}$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency | Found Index | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.091000s | 0.30x | 30.0% | 50000000 | PASSED |
| **2** | 0.043000s | 0.63x | 31.5% | 50000000 | PASSED |
| **4** | 0.030000s | 0.90x | 22.5% | 50000000 | PASSED |
| **8** | 0.021000s | 1.29x | 16.1% | 50000000 | PASSED |


## 3. Analysis & Discussion

### How is the search parallelized and how are race conditions avoided?

- **Domain Decomposition:** The array iteration space $[0, N-1]$ is statically partitioned into equal-sized chunks across worker threads.
- **Race Condition Prevention:** When a thread locates the target element, it enters a `#pragma omp critical` section to compare and update the shared `found_idx` variable. This guarantees that if multiple duplicate targets exist, the smallest (first) matching array index is deterministically recorded without data races.

### Early Termination & Thread Coordination:

- **Flag-Based Short Circuiting:** A shared `found_idx` flag is inspected during iteration. Once any thread finds the target element and sets `found_idx`, other threads processing higher index ranges immediately skip remaining loop iterations via the `continue` check, preventing unnecessary memory scans.
- **Memory Bandwidth Bounds:** Array search involves simple comparison operations ($O(N)$ reads with zero arithmetic operations per element). Performance scaling is bound by DRAM memory read bandwidth when searching large arrays ($N \ge 10^7$).