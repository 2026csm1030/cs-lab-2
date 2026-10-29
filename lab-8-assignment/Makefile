CC = gcc
CFLAGS = -fopenmp -O2 -Wall
LDFLAGS = -lm

.PHONY: all task1 task2 task3 task4 task5 clean

all: task1 task2 task3 task4 task5

task1: task1/task1_search.c
	$(CC) $(CFLAGS) task1/task1_search.c -o task1/task1_runner

task2: task2/task2_sort.c
	$(CC) $(CFLAGS) task2/task2_sort.c -o task2/task2_runner

task3: task3/task3_histogram.c
	$(CC) $(CFLAGS) task3/task3_histogram.c -o task3/task3_runner

task4: task4/task4_transpose.c
	$(CC) $(CFLAGS) task4/task4_transpose.c -o task4/task4_runner

task5: task5/task5_conv2d.c
	$(CC) $(CFLAGS) task5/task5_conv2d.c $(LDFLAGS) -o task5/task5_runner

clean:
	rm -f task1/task1_runner task2/task2_runner task3/task3_runner task4/task4_runner task5/task5_runner