# Task 1: Parallel Merge Sort using OpenMP Tasks — Report

## 1. System Configuration & Task Constructs

- **OpenMP Directives:** `#pragma omp task`, `#pragma omp taskwait`, `#pragma omp single` inside `#pragma omp parallel`.
- **Cutoff Threshold:** $2048$ elements (`right - left < 2048` switches to `merge_sort_serial`).
- **Compiler Flags:** `gcc -fopenmp -O2 task1_mergesort.c -o task1_runner`.
- **Verification:** All parallel outputs were verified element-by-element against serial reference outputs for exact sorting order.

## 2. Benchmark Results

### Execution Time Table (seconds)

| Array Size ($N$) | Total Memory (MB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | 0.38 MB | 0.012000s | 0.016000s | 0.008000s | 0.006000s | 0.004000s |
| **$10^6$** | 3.81 MB | 0.139000s | 0.120000s | 0.062000s | 0.045000s | 0.030000s |
| **$10^7$** | 38.15 MB | 1.326000s | 1.294000s | 0.683000s | 0.443000s | 0.287000s |


### Speedup and Parallel Efficiency Table ($N = 10^6$, 3.81 MB Array)

> **Serial Baseline Time ($T_{\text{serial}}$):** $0.139000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.120000s | 1.16x | 115.8% | PASSED |
| **2** | 0.062000s | 2.24x | 112.1% | PASSED |
| **4** | 0.045000s | 3.09x | 77.2% | PASSED |
| **8** | 0.030000s | **4.63x** | 57.9% | PASSED |


### Speedup and Parallel Efficiency Table ($N = 10^7$, 38.15 MB Array)

> **Serial Baseline Time ($T_{\text{serial}}$):** $1.326000\text{ seconds}$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 1.294000s | 1.02x | 102.5% | PASSED |
| **2** | 0.683000s | 1.94x | 97.1% | PASSED |
| **4** | 0.443000s | 2.99x | 74.8% | PASSED |
| **8** | 0.287000s | **4.62x** | 57.8% | PASSED |

## 3. Analysis & Questions

### Q1: What is the difference between OpenMP tasks and `parallel for`?
- `#pragma omp parallel for` is intended for **loop-level parallelism** over countable, static loop iteration bounds where iterations are split evenly across threads up front.
- `#pragma omp task` is designed for **unstructured or recursive parallelism** (e.g., recursive divide-and-conquer algorithms, irregular tree traversals). Each task represents an independent unit of work packaged dynamically at runtime and assigned asynchronously to available worker threads from an OpenMP task queue.

### Q2: Why is `#pragma omp taskwait` required before merging the two halves?
- Merge sort requires that both the left subarray `a[left..mid]` and right subarray `a[mid+1..right]` are fully sorted before calling `merge(a, tmp, left, mid, right)`.
- Because `#pragma omp task` executes child tasks asynchronously in background threads, the parent thread would immediately invoke `merge()` while sorting is still in progress. `#pragma omp taskwait` forces the parent task to suspend until all child tasks spawned in its immediate scope complete, guaranteeing strict correctness.

### Q3: How does the cutoff threshold affect performance?
- **Without Cutoff (Threshold = 1):** Spawning OpenMP tasks down to 1-element arrays creates millions of micro-tasks. The overhead of task allocation, stack creation, scheduling, and synchronization overwhelms actual computational work, leading to severe slowdowns.
- **With Cutoff (Threshold = 2048):** Subarrays smaller than 2048 elements are sorted using fast serial merge sort. This avoids micro-task management overhead while keeping enough coarse-grained tasks to achieve up to **4.63x speedup** on 8 threads.

### Q4: Why can parallel merge sort become slower than serial merge sort for small arrays?
- For small array sizes ($N = 10^5$), 1-thread execution ($0.016\text{s}$) is slower than serial baseline ($0.012\text{s}$, $0.75\text{x}$ speedup) because thread-pool creation and task-queue overhead outweigh the lightweight sorting work.

### Q5: How does task creation overhead affect scalability?
- Task creation and scheduling incur fixed runtime overhead per task. Coarse-grained task decomposition (via cutoff thresholds) reduces overall task count, maintaining high parallel efficiency ($57.8\%$ to $112.1\%$) across thread counts.