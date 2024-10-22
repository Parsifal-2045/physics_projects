/*

This program will numerically compute the integral of

				  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.

*/

#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

#define MAX_N_THREADS 2

int main()
{
	double start_time, run_time;

	step = 1.0 / (double)num_steps;

	double sum[MAX_N_THREADS];

	omp_set_num_threads(MAX_N_THREADS);
	int n_threads = 0;

	start_time = omp_get_wtime();

#pragma omp parallel
	{
		double x;
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

		if (tid == 0)
		{
			n_threads = num_threads;
		}

		int i;
		for (i = tid, sum[tid] = 0.0; i <= num_steps; i += num_threads)
		{
			x = (i + 0.5) * step;
			sum[tid] += 4.0 / (1.0 + x * x);
		}
	}

	double pi = 0.0;
	for (int i = 0; i != n_threads; i++)
	{
		pi += step * sum[i];
	}

	run_time = omp_get_wtime() - start_time;
	printf("\n pi with %ld steps is %lf in %lf seconds\n ", num_steps, pi, run_time);
}