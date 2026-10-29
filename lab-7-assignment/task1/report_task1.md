# Task 1: Vector Addition — Report

## 1. System Configuration

- **Number of Cores:** 6 Cores
- **Compiler:** `gcc -fopenmp -O2 task1_vecadd.c -o task1`
- **Thread Management:** Explicitly configured via OpenMP `num_threads` clause / runtime environment variable (`export OMP_NUM_THREADS`).

## 2. Benchmark Results

### Execution Time Table (seconds)
| Array Size ($N$) | Total Memory (MB) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | 2.29 MB | 0.001000s | 0.000000s | 0.001000s | 0.001000s | 0.000000s |
| **$10^6$** | 22.89 MB | 0.005000s | 0.005000s | 0.003000s | 0.004000s | 0.003000s |
| **$10^7$** | 228.88 MB | 0.049000s | 0.056000s | 0.024000s | 0.031000s | 0.026000s |
| **$10^8$** | 2288.82 MB | 0.406000s | 0.422000s | 0.100000s | 0.099000s | 0.113000s |

### Speedup Table for $N = 10^8$ ($S = T_{\text{serial}} / T_{\text{parallel}}$)
| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) |
| :---: | :---: | :---: | :---: |
| **1** | 0.422000s | 0.96x | 96.2% |
| **2** | 0.100000s | 4.06x | 203.0% |
| **4** | 0.099000s | 4.10x | 102.5% |
| **8** | 0.113000s | 3.59x | 44.9% |

## 3. Analysis & Questions

### Q1: At what array size does parallelization actually start paying off? Why is it not beneficial for small $N$?

- **Payoff Threshold:** Parallelization usually becomes beneficial around $N = 10^6$ to $10^7$ elements.
- **Explanation:** For small $N$ (e.g., $10^5$), the total computational load is extremely low ($O(N)$ operations). The overhead associated with spawning, synchronizing, and distributing loop iterations across OpenMP worker threads exceeds the execution time of the loop itself. As $N$ grows, the computation dominates thread management overhead.

### Q2: Does speedup keep increasing linearly with thread count? Why or why not?

- **Answer:** No, speedup plateaus sub-linearly and does not achieve ideal linear scaling.
- **Explanation:** Vector addition is fundamentally a **memory-bandwidth bound** operation rather than a CPU-bound operation. It has a low arithmetic intensity (only 1 floating-point addition for every 2 vector reads and 1 vector write). Once the memory bus bandwidth between the CPU cache and DRAM saturates (typically around 2 to 4 cores), adding more threads cannot fetch data faster, causing the speedup curve to flatten.

