# Task 2: Sum of Array Elements — Report

## 1. System Configuration

- **Number of Cores:** 6 Cores
- **Compiler Flags:** `gcc -fopenmp -O2 task2_arraysum.c -o task2_runner`
- **Thread Management:** Explicitly configured via OpenMP `num_threads` clause / runtime environment variable (`export OMP_NUM_THREADS`).

## 2. Benchmark Results

### Demonstration of Naive Unsafe Parallel Sum ($N = 10^7$, 4 Threads)
| Run # | Calculated Sum | Expected Sum | Status |
| :---: | :---: | :---: | :---: |
| **1** | 2500000 | 10000000 | DATA RACE |
| **2** | 2500000 | 10000000 | DATA RACE |
| **3** | 2500000 | 10000000 | DATA RACE |
| **4** | 2500000 | 10000000 | DATA RACE |
| **5** | 2500000 | 10000000 | DATA RACE |

---

### Performance Comparison: `reduction(+:sum)`

#### Execution Time Table (seconds)
| Array Size ($N$) | Serial Time | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^7$** | 0.006000s | 0.006000s | 0.004000s | 0.004000s | 0.015000s |
| **$10^8$** | 0.052000s | 0.055000s | 0.040000s | 0.035000s | 0.037000s |

#### Speedup Table for $N = 10^8$ ($S = T_{\text{serial}} / T_{\text{parallel}}$)

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.055000s | 0.95x | 95.0% | PASSED |
| **2** | 0.040000s | 1.30x | 65.0% | PASSED |
| **4** | 0.035000s | 1.49x | 37.3% | PASSED |
| **8** | 0.037000s | 1.41x | 17.6% | PASSED |

### Critical Section vs. Reduction Performance ($N = 10^7$)
| Thread Count ($T$) | Reduction Time (s) | Critical Time (s) | Overhead Factor ($T_{\text{critical}} / T_{\text{reduction}}$) |
| :---: | :---: | :---: | :---: |
| **1** | 0.006000s | 0.094000s | 15.67x slower |
| **2** | 0.004000s | 0.156000s | 39.00x slower |
| **4** | 0.004000s | 0.213000s | 53.25x slower |
| **8** | 0.015000s | 0.407000s | 27.13x slower |

## 3. Analysis & Questions

### Q1: Why does the naive version give wrong/inconsistent answers? Explain in terms of a data race on the shared variable `sum`.

- **Explanation:** In the naive implementation, multiple threads concurrently execute `sum += a[i]` on a single shared variable `sum`. At the assembly level, `sum += a[i]` consists of three non-atomic steps:
  1. Read `sum` from memory into a CPU register.
  2. Add `a[i]` to the register value.
  3. Write the register value back to memory.
- When multiple threads execute these steps simultaneously without synchronization, interleaving occurs. Thread B may overwrite Thread A's write, resulting in lost updates and non-deterministic, incorrect results across runs.

### Q2: Would using `#pragma omp critical` around `sum += a[i]` fix correctness? What happens to performance compared to `reduction`?

- **Correctness:** Yes, placing `#pragma omp critical` around `sum += a[i]` enforces mutual exclusion. Only one thread updates `sum` at any given time, eliminating data races and restoring exact arithmetic correctness.
- **Performance:** Performance degrades drastically (often 100x to 1000x slower than serial). Every single iteration forces thread serialization, lock acquisition, bus locking, and cache line bouncing on `sum`.
- **Why `reduction` is Superior:** The `reduction(+:sum)` clause allocates a private accumulator variable for each thread. Threads accumulate sums locally in L1 cache with zero synchronization overhead during loop execution. When the loop finishes, OpenMP combines the local sums into the global `sum` in a single serial step.


