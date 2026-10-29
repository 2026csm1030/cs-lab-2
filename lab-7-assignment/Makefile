CC = gcc
CFLAGS = -fopenmp -O2 -Wall
LDFLAGS = -lm

.PHONY: all task1 task2 task3 task4 task5 clean

all: task1 task2 task3 task4 task5

task1: task1/task1_vecadd.c
	$(CC) $(CFLAGS) task1/task1_vecadd.c -o task1/task1_runner

task2: task2/task2_arraysum.c
	$(CC) $(CFLAGS) task2/task2_arraysum.c -o task2/task2_runner

task3: task3/task3_maxidx.c
	$(CC) $(CFLAGS) task3/task3_maxidx.c -o task3/task3_runner

task4: task4/task4_matadd.c
	$(CC) $(CFLAGS) task4/task4_matadd.c -o task4/task4_runner

task5: task5/task5_matmul.c
	$(CC) $(CFLAGS) task5/task5_matmul.c $(LDFLAGS) -o task5/task5_runner

clean:
	rm -f task1/task1_runner task2/task2_runner task3/task3_runner task4/task4_runner task5/task5_runner