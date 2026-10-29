# Task 5: 2D Convolution — Report

## 1. System Configuration & Problem Setup

- **Kernel / Filter:** $3 \times 3$ Averaging Kernel ($K_{i,j} = \frac{1}{9}$).
- **Output Dimensions:** For an input matrix of $H \times W$, valid convolution without padding yields output dimensions of $(H - 2) \times (W - 2)$.
- **Compiler Flags:** `gcc -fopenmp -O2 task5_conv2d.c -o task5_runner -lm`
- **Thread Control:** OpenMP `num_threads` clause.

## 2. Sample Assignment Verification ($5 \times 5$ Image)

### Input Image Matrix ($5 \times 5$)
$$\begin{bmatrix} 10 & 20 & 30 & 40 & 50 \\ 60 & 70 & 80 & 90 & 100 \\ 110 & 120 & 130 & 140 & 150 \\ 160 & 170 & 180 & 190 & 200 \\ 210 & 220 & 230 & 240 & 250 \end{bmatrix}$$

### Computed Output Matrix ($3 \times 3$)
$$\begin{bmatrix} 70.00 & 80.00 & 90.00 \\ 120.00 & 130.00 & 140.00 \\ 170.00 & 180.00 & 190.00 \end{bmatrix}$$

- **Center Sample Verification:** $O[1][1] = \frac{10+20+30+60+70+80+110+120+130}{9} = \frac{630}{9} = 70.00$
- **Verification Status:** **PASSED** (Serial and Parallel 4-thread outputs matched expected matrix calculations exactly).

## 3. Benchmark Results

### Method 1: Outermost Loop Parallelization (`#pragma omp parallel for`)

#### Execution Time Table (seconds)
| Image Size ($N \times N$) | Memory (MB) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1024 \times 1024$** | 23.94 MB | 0.006000s | 0.007000s | 0.004000s | 0.002000s | 0.003000s |
| **$2048 \times 2048$** | 95.88 MB | 0.021000s | 0.023000s | 0.013000s | 0.008000s | 0.005000s |
| **$4096 \times 4096$** | 383.75 MB | 0.084000s | 0.089000s | 0.039000s | 0.028000s | 0.016000s |

#### Speedup & Efficiency Table for Largest Image ($4096 \times 4096$)
> **Serial Baseline Time ($T_{\text{serial}}$):** $0.084000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Execution Time (s) | Speedup ($S$) | Parallel Efficiency ($S / T$) | Verification |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.089000s | 0.94x | 94.4% | PASSED |
| **2** | 0.039000s | 2.15x | 107.7% | PASSED |
| **4** | 0.028000s | 3.00x | 75.0% | PASSED |
| **8** | 0.016000s | **5.25x** | 65.6% | PASSED |

### Method 2: Collapsed Loops (`#pragma omp parallel for collapse(2)`)

| Image Size ($N \times N$) | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1024 \times 1024$** | 0.006000s | 0.006000s | 0.005000s | 0.002000s | 0.002000s | 3.00x |
| **$2048 \times 2048$** | 0.021000s | 0.025000s | 0.010000s | 0.009000s | 0.006000s | 3.50x |
| **$4096 \times 4096$** | 0.084000s | 0.084000s | 0.061000s | 0.032000s | 0.019000s | **4.42x** |

### Method 1 vs Method 2 Comparison ($4096 \times 4096$)

| Thread Count ($T$) | Outer Loop Time (s) | Outer Loop Speedup | Collapse(2) Time (s) | Collapse(2) Speedup | Performance Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.089000s | 0.94x | 0.084000s | 1.00x | Collapse (+6.0%) |
| **2** | 0.039000s | 2.15x | 0.061000s | 1.38x | Outer Loop (+55.8%) |
| **4** | 0.028000s | 3.00x | 0.032000s | 2.62x | Outer Loop (+14.3%) |
| **8** | 0.016000s | **5.25x** | 0.019000s | 4.42x | Outer Loop (+18.8%) |

## 4. Analysis & Discussion

### Parallelization Safety & Zero Race Conditions

- In 2D convolution, each output cell $O[i][j]$ is calculated independently by evaluating a $3 \times 3$ neighborhood of the input image $I$.
- No output cell depends on the result of any other output cell. Because worker threads write to non-overlapping, disjoint memory indices in $O$, there are zero data races, making `#pragma omp parallel for` completely safe without locks or synchronization.

### Performance & Scaling Comparison: Outer Loop vs. `collapse(2)`

- **Outer Loop Performance:** Parallelizing only the outer row loop $i$ ($4094$ row iterations for $N=4096$) provides coarse, contiguous chunks of rows to each OpenMP thread. At 8 threads, execution time drops from $0.084\text{s}$ down to $0.016\text{s}$ (**5.25x speedup**).
- **Why `collapse(2)` Performs Worse at Large Sizes:** Collapsing $i$ and $j$ loops creates $4094 \times 4094 \approx 1.67 \times 10^7$ virtual iterations. This introduces subtle index division/modulo arithmetic overhead when mapping flattened loop indices back to 2D coordinates `(i, j)`. Since $4094$ outer rows already provide more than enough tasks to saturate 8 threads, collapsing loops adds unnecessary scheduling overhead, resulting in a lower maximum speedup ($4.42\text{x}$ vs $5.25\text{x}$).

### Cache Locality & Operational Intensity

- As the $3 \times 3$ filter slides horizontally across adjacent columns $j$, 6 of the 9 loaded input pixel values are already present in L1/L2 cache lines from the previous iteration.
- This high degree of spatial locality and input pixel reuse allows 2D convolution to achieve better parallel scaling ($5.25\text{x}$ speedup at 8 threads) than purely memory-bandwidth-bound operations like vector addition[cite: 1, 2].