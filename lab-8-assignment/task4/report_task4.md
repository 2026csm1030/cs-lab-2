# Task 4: Matrix Transpose using OpenMP — Report

## 1. System Configuration

- **Memory Representation:** Contiguous 1D row-major array indexing (`[i * N + j]` for row $i$, column $j$).
- **Compiler Flags:** `gcc -fopenmp -O2 task4_transpose.c -o task4_runner`.
- **Verification:** Element-wise exact comparison against the serial matrix transpose output.

## 2. Benchmark Results

### Method 1: Outer Loop Parallelization (`#pragma omp parallel for`)

#### Execution Time Table (seconds)
| Matrix Size ($N \times N$) | Total Memory (MB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1024 \times 1024$** | 16.00 MB | 0.013000s | 0.012000s | 0.008000s | 0.002000s | 0.003000s |
| **$2048 \times 2048$** | 64.00 MB | 0.057000s | 0.056000s | 0.022000s | 0.011000s | 0.009000s |
| **$4096 \times 4096$** | 256.00 MB | 0.231000s | 0.228000s | 0.088000s | 0.042000s | 0.029000s |

### Method 2: Collapsed Loops (`#pragma omp parallel for collapse(2)`)

| Matrix Size ($N \times N$) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1024 \times 1024$** | 0.013000s | 0.009000s | 0.004000s | 0.002000s | 0.002000s | 6.50x |
| **$2048 \times 2048$** | 0.057000s | 0.044000s | 0.021000s | 0.013000s | 0.010000s | 5.70x |
| **$4096 \times 4096$** | 0.231000s | 0.174000s | 0.088000s | 0.042000s | 0.030000s | 7.70x |

### Method 3: Parallel Tiled Transpose (32x32 Tiles)

#### Execution Time & Speedup Comparison ($N = 4096 \times 4096$)
> **Serial Non-Tiled Baseline:** $0.231000\text{ seconds}$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.082000s | 2.82x | 281.7% | PASSED |
| **2** | 0.044000s | 5.25x | 262.5% | PASSED |
| **4** | 0.027000s | 8.56x | 213.9% | PASSED |
| **8** | 0.022000s | 10.50x | 131.2% | PASSED |

## 3. Analysis & Discussion

### Memory Locality Bottleneck in Standard Transpose:

- In a naive matrix transpose $B[j][i] = A[i][j]$, reads from matrix $A$ proceed linearly along rows (high spatial locality), but writes into matrix $B$ jump vertically across memory columns with a stride of $N \times \text{sizeof(double)}$.
- For large matrices ($N = 4096$, $256\text{ MB}$ data size), stride-$N$ memory writes cause constant L1/L2 cache line thrashing and cache evictions, making memory bus bandwidth the primary bottleneck.

### Performance Impact of Cache Tiling ($32 \times 32$ Blocks):

- **Superlinear Single-Thread Speedup:** Even on 1 thread, the tiled algorithm reduces execution time from **$0.231\text{s}$** to **$0.082\text{s}$** (**2.82x faster** than non-tiled serial code). This efficiency gain stems from keeping active sub-blocks inside the fast CPU cache, preventing cache misses.
- **Parallel Scaling Advantage:** At 8 threads, tiled transpose achieves a peak speedup of **10.50x** ($0.022\text{s}$) compared to non-tiled outer loop parallelization (**7.97x**, $0.029\text{s}$). Restructuring memory accesses to respect cache locality allows OpenMP threads to scale past memory bus saturation limits.