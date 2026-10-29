# OpenMP | CS

https://www.youtube.com/watch?v=dcT7y6kLwMs

Can tell which variable is private and which is shared:

```cpp
int x, y, x;

#pragma omp parallel private(x) shared(y, x)
```

To define the critical section for when only one thread will enter the given block or section and will completely finish before any other thread can come to this section:

```cpp
#pragma omp parallel
{
		// parallel code..
		
		#pragma omp critical
		{
				cout << omp_get_thread_num() << "\n";
		}
}
```

Use `reduction` to manage the shared variable or memory among the threads.

```cpp
int sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++)
{
	sum += // some parallel work
}
```

Absolutely. Since this is your **first time learning OpenMP**, let's start from zero and build up to:

```
C = A + B
```

in parallel.

---

# 1. First: What problem are we trying to solve?

Suppose:

```
A = [1, 2, 3, 4]
B = [10, 20, 30, 40]
```

We want:

```
C[0] = A[0] + B[0] = 11
C[1] = A[1] + B[1] = 22
C[2] = A[2] + B[2] = 33
C[3] = A[3] + B[3] = 44
```

In C++:

```
for (int i = 0; i < 4; i++) {
    C[i] = A[i] + B[i];
}
```

Normally, this is sequential:

```
i = 0 → C[0]
i = 1 → C[1]
i = 2 → C[2]
i = 3 → C[3]
```

But notice something important:

**None of these calculations depends on another.**

```
C[0] = A[0] + B[0]   independent
C[1] = A[1] + B[1]   independent
C[2] = A[2] + B[2]   independent
C[3] = A[3] + B[3]   independent
```

Therefore, we can give different iterations to different threads.

```
Thread 1 → C[0]
Thread 2 → C[1]
Thread 3 → C[2]
Thread 4 → C[3]
```

That's exactly the kind of problem OpenMP is useful for.

---

# 2. What exactly is OpenMP?

**OpenMP is an API/programming model for writing shared-memory parallel programs in C, C++, and Fortran.**

You don't have to manually create and manage CPU threads yourself.

Instead, you give OpenMP instructions such as:

> "This loop can be executed by multiple threads."
> 

OpenMP then handles the thread creation/distribution for you.

The important thing to understand is:

```
Your C++ program
       ↓
      OpenMP
       ↓
Multiple threads
       ↓
CPU cores
```

---

# 3. Our normal C++ program

Let's first write the sequential version.

```
#include <iostream>

int main() {

    const int N = 4;

    int A[N] = {1, 2, 3, 4};
    int B[N] = {10, 20, 30, 40};
    int C[N];

    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }

    return 0;
}
```

Output:

```
11 22 33 44
```

Nothing parallel yet.

---

# 4. Now introduce OpenMP

First, include the OpenMP header:

```
#include <omp.h>
```

Then we can tell OpenMP:

```
#pragma omp parallel for
```

So our loop becomes:

```
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

That's it.

This is one of the most important things to understand about OpenMP:

> **A `#pragma omp` directive tells the compiler/OpenMP runtime how a particular piece of code should be parallelized.**
> 

---

# 5. Let's understand `parallel for`

This:

```
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

can be thought of as:

```
parallel
   +
for
```

### `parallel`

Create a group of threads that can execute work concurrently.

### `for`

Distribute the iterations of the following loop among those threads.

So if we have:

```
N = 8
```

and:

```
4 threads
```

OpenMP might distribute the work roughly like:

```
Thread 0 → i = 0, 1
Thread 1 → i = 2, 3
Thread 2 → i = 4, 5
Thread 3 → i = 6, 7
```

Conceptually:

```
                 for loop
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Thread 0    Thread 1    Thread 2 ...
       │            │            │
      C[0]         C[2]         C[4]
      C[1]         C[3]         C[5]
```

The exact distribution can depend on the scheduling configuration.

---

# 6. Complete OpenMP example

```
#include <iostream>
#include <omp.h>

int main() {

    const int N = 8;

    int A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    int C[N];

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }

    return 0;
}
```

Output:

```
11 22 33 44 55 66 77 88
```

The output is the same as the sequential version.

The **difference is how the work is performed**.

---

# 7. Where do the threads come from?

Suppose your computer has multiple CPU cores.

OpenMP can create multiple **threads**.

For example:

```
             CPU
    ┌────────┼────────┐
    ↓        ↓        ↓
 Core 1    Core 2    Core 3    Core 4
    ↑        ↑        ↑        ↑
 Thread 0  Thread 1 Thread 2 Thread 3
```

OpenMP manages these threads for you.

You don't need to write:

```
createThread();
startThread();
joinThread();
```

yourself.

That's one of the major reasons OpenMP is convenient.

---

# 8. How many threads does OpenMP use?

You can explicitly specify the number of threads.

For example:

```
#include <omp.h>

omp_set_num_threads(4);
```

Then:

```
omp_set_num_threads(4);

#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

OpenMP will use **4 threads** for the parallel region.

Alternatively, you can use an environment variable:

```
OMP_NUM_THREADS=4
```

This is particularly useful when you're experimenting with performance.

---

# 9. How do we know that multiple threads are actually being used?

You can print the thread ID.

For example:

```
#pragma omp parallel for
for (int i = 0; i < N; i++) {

    int thread_id = omp_get_thread_num();

    std::cout << "Thread "
              << thread_id
              << " calculates C[" << i << "]\n";

    C[i] = A[i] + B[i];
}
```

You might see something like:

```
Thread 0 calculates C[0]
Thread 0 calculates C[1]
Thread 1 calculates C[2]
Thread 1 calculates C[3]
Thread 2 calculates C[4]
Thread 2 calculates C[5]
Thread 3 calculates C[6]
Thread 3 calculates C[7]
```

**But don't rely on the order of these printed lines.**

Threads execute concurrently, so output ordering isn't guaranteed.

---

# 10. Why is this particular loop safe?

This is one of the most important concepts for your GEMM experiment later.

Our calculation is:

```
C[i] = A[i] + B[i];
```

Suppose:

```
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
Thread 3 → C[3]
```

Each thread writes to a **different element of C**.

Nobody is doing:

```
Thread 0 → C[0]
Thread 1 → C[0]
```

So there is no conflict.

Also, `A` and `B` are only being read.

Therefore this is a very good loop to parallelize.

---

# 11. The important rule

Before parallelizing a loop, ask:

> **Can different iterations execute independently?**
> 

For:

```
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

Yes.

Because:

```
iteration 0 doesn't depend on iteration 1
iteration 1 doesn't depend on iteration 2
iteration 2 doesn't depend on iteration 3
...
```

Therefore:

```
#pragma omp parallel for
```

is appropriate.

---

# 12. Compare sequential vs OpenMP

### Sequential

```
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

```
Thread
  │
  ├── C[0]
  ├── C[1]
  ├── C[2]
  ├── C[3]
  └── ...
```

### OpenMP

```
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

```
             loop
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
    Thread  Thread  Thread
      0       1       2
       ↓      ↓      ↓
     C[0]   C[1]   C[2]
     C[3]   C[4]   C[5]
       ...
```

---

# 13. Compiling an OpenMP program

This part is important because simply writing:

```
#include <omp.h>
```

isn't enough.

If you're using **GCC/G++**, compile with:

```
g++ -fopenmp program.cpp -o program
```

The important flag is:

```
-fopenmp
```

It tells the compiler to enable OpenMP.

Then:

```
./program
```

On Windows with MinGW/GCC, the same `-fopenmp` option applies.

---

# 14. One subtle but important point

Parallel doesn't automatically mean **faster**.

Suppose:

```
N = 4
```

Creating/managing multiple threads may cost more time than simply doing four additions.

OpenMP becomes more useful when the amount of work is sufficiently large.

That's why your professor's **GEMM experiment** is a good example.

For:

```
N = 10
```

there isn't much work.

For:

```
N = 1000
```

there are:

\[
1000^3 = 1,000,000,000
\]

multiplications/additions involved in the naïve GEMM.

Now parallelism becomes much more meaningful.

---

# 15. The connection to GEMM

Now you should be able to see why your professor suggested:

> **"Try using OpenMP to run GEMM in parallel."**
> 

GEMM:

```
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = 0;

        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

The outer work is:

```
C[0][0]
C[0][1]
C[0][2]
...
C[1][0]
C[1][1]
...
```

These calculations can be distributed across threads, just like:

```
C[0]   C[1]   C[2]   C[3]
```

were distributed in our array-addition example.

So **array addition is the perfect first OpenMP example**, and GEMM is essentially the next level.

### What I recommend you learn next

Before jumping into GEMM, make sure these **5 OpenMP concepts** are clear:

1. `#pragma omp parallel`
2. `#pragma omp parallel for`
3. Number of threads
4. Thread IDs
5. **Shared vs private variables / race conditions**

Once those are clear, the GEMM code will make much more sense rather than feeling like we're just adding a `#pragma` and hoping it works.