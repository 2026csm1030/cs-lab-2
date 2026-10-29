CC = gcc
CFLAGS = -fopenmp -O2 -Wall

.PHONY: all task1 task2 task3 clean

all: task1 task2 task3

task1: task1/task1_mergesort.c
	$(CC) $(CFLAGS) task1/task1_mergesort.c -o task1/task1_runner

task2: task2/task2_scheduling.c
	$(CC) $(CFLAGS) task2/task2_scheduling.c -o task2/task2_runner

task3: task3/task3_even_odd.c
	$(CC) $(CFLAGS) task3/task3_even_odd.c -o task3/task3_runner

clean:
	rm -f task1/task1_runner task2/task2_runner task3/task3_runner