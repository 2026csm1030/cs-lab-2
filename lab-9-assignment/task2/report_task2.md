# Task 2: OpenMP Scheduling Strategies (Uneven Workloads) — Report

## 1. System Configuration
- **Workload Simulation:** Simulated uneven loop iteration execution time via `work = (i % 1000) * factor`.
- **Compiler Flags:** `gcc -fopenmp -O2 task2_scheduling.c -o task2_runner`
- **Thread Management:** OpenMP `num_threads` directive clause.
- **Verification Status:** All parallel scheduling configurations verified exact numerical results matching the serial baseline.

## 2. Benchmark Results

### Execution Time Table (seconds) Across Scheduling Strategies

| $N$ | Strategy | Serial Baseline | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Max Speedup (8T) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^5$** | `static` | 0.600000s | 0.605000s | 0.306000s | 0.153000s | 0.082000s | 7.32x |
| **$10^5$** | `dynamic, 100` | 0.600000s | 0.602000s | 0.304000s | 0.175000s | 0.076000s | **7.89x** |
| **$10^5$** | `guided` | 0.600000s | 0.598000s | 0.303000s | 0.153000s | 0.085000s | 7.06x |
| | | | | | | | |
| **$10^6$** | `static` | 4.670000s | 4.429000s | 2.184000s | 1.091000s | 0.578000s | 8.08x |
| **$10^6$** | `dynamic, 100` | 4.670000s | 4.364000s | 2.186000s | 1.094000s | 0.566000s | 8.25x |
| **$10^6$** | `guided` | 4.670000s | 4.379000s | 2.203000s | 1.089000s | 0.544000s | **8.58x** |
| | | | | | | | |
| **$10^7$** | `static` | 44.017000s | 44.135000s | 21.740000s | 10.847000s | 5.451000s | 8.08x |
| **$10^7$** | `dynamic, 100` | 44.017000s | 43.804000s | 21.762000s | 10.849000s | 5.413000s | **8.13x** |
| **$10^7$** | `guided` | 44.017000s | 43.848000s | 21.799000s | 10.881000s | 5.425000s | 8.11x |

### Speedup and Parallel Efficiency Summary ($N = 10^7$)

> **Serial Baseline Time:** $44.017000\text{ seconds}$  
> Speedup ($S$) = $T_{\text{serial}} / T_{\text{parallel}}$, Parallel Efficiency = $(S / T) \times 100\%$

| Thread Count ($T$) | Best Execution Time (s) | Strategy | Speedup ($S$) | Parallel Efficiency | Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 43.804000s | `dynamic, 100` | 1.00x | 100.5% | PASSED |
| **2** | 21.740000s | `static` | 2.02x | 101.2% | PASSED |
| **4** | 10.847000s | `static` | 4.06x | 101.4% | PASSED |
| **8** | 5.413000s | `dynamic, 100` | **8.13x** | **101.6%** | PASSED |

## 3. Analysis & Questions

### Q1: What is the difference between static, dynamic, and guided scheduling?
- **`schedule(static)`:** Iterations are divided into equal chunk sizes at loop initialization and assigned statically to threads in round-robin order up front. It incurs zero runtime scheduling overhead, but can cause load imbalance if work per iteration varies unpredictably.
- **`schedule(dynamic, chunk_size)`:** Iterations are dynamically allocated to worker threads from a global shared work queue in chunks of size `chunk_size`. As soon as a thread finishes its current chunk, it requests another chunk from the runtime. This prevents thread idling on highly irregular workloads at the cost of runtime synchronization overhead.
- **`schedule(guided)`:** Iterations are dynamically assigned in variable chunk sizes that start large (proportional to remaining unassigned iterations) and shrink exponentially down to a minimum chunk size. This minimizes thread synchronization overhead during early iterations while preserving fine-grained load balancing toward the end.

### Q2: Which scheduling strategy performs best for uniform workloads? What about uneven workloads?
- **Uniform Workloads:** **`static` scheduling** performs best because every iteration requires identical execution time, enabling static partitioning to achieve ideal load balance without paying runtime work-queue locks.
- **Uneven Workloads:** **`dynamic` and `guided` scheduling** perform best on irregular workloads. In our benchmark, `guided` achieved the highest peak speedup of **8.58x** at 8 threads for $N = 10^6$ ($0.544\text{s}$), and `dynamic, 100` achieved **8.13x** for $N = 10^7$ ($5.413\text{s}$). Because `work = (i % 1000) * factor` exhibits a repeating periodic pattern, static partitioning also divides work fairly evenly at large $N$ ($8.08\text{x}$ speedup).

### Q3: Why can dynamic scheduling introduce additional overhead?
- Dynamic scheduling requires thread synchronization (internal lock/atomic operations) every time a worker thread completes a chunk and queries the global queue for its next assignment. If chunk sizes are too small or problem sizes are small, threads spend more CPU cycles competing for queue locks than doing actual computation[cite: 5].

### Q4: How does the chunk size affect performance?
- **Small Chunk Size (e.g., `chunk_size = 1`):** Provides fine-grained load distribution, but creates severe lock contention and queue management overhead, slowing down execution[cite: 5].
- **Large Chunk Size (e.g., `chunk_size = 1000+`):** Reduces runtime locking overhead, but begins to resemble static scheduling, increasing the risk of load imbalance if long iterations are clustered inside a single chunk.
- **Optimal Choice (`chunk_size = 100` or `guided`):** Strikes an ideal balance between low runtime synchronization overhead and adaptive load distribution across CPU cores.