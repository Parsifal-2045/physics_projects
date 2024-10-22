/*
**  PROGRAM: Mandelbrot area
**
**  PURPOSE: Program to compute the area of a  Mandelbrot set.
**           Correct answer should be around 1.510659.
**           WARNING: this program may contain errors
**
**  USAGE:   Program runs without input ... just run the executable
**
**  HISTORY: Written:  (Mark Bull, August 2011).
**           No structures or file-scope issues (T. Mattson, Nov. 2023)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NPOINTS 1000
#define MAXITER 1000

void testPoint(const double, const double);

int numoutside = 0;

int main()
{
  // Loop over grid of points in the complex plane which contains the Mandelbrot
  // set, testing each point to see whether it is inside or outside the set.
#pragma omp parallel for
  for (int i = 0; i < NPOINTS; ++i)
  {
    for (int j = 0; j < NPOINTS; ++j)
    {
      const double epsilon = 1.0e-5; // private copy per thread
      const double creal = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + epsilon;
      const double cimag = 1.125 * (double)(j) / (double)(NPOINTS) + epsilon;
      testPoint(creal, cimag);
    }
  }

  // Calculate area of set and error estimate. Output the results
  double area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
  double error = area / (double)NPOINTS;

  printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
  printf("Correct answer should be around 1.510659\n");
}

void testPoint(const double creal, const double cimag)
{
  // Iterate z=z*z+c, until |z| > 2 when point is known to be outside set
  // If loop count reaches MAXITER, point is considered to be inside the set
  double zreal, zimag, temp;
  zreal = creal;
  zimag = cimag;

  for (int iter = 0; iter < MAXITER; ++iter)
  {
    temp = (zreal * zreal) - (zimag * zimag) + creal;
    zimag = zreal * zimag * 2 + cimag;
    zreal = temp;
    if ((zreal * zreal + zimag * zimag) > 4.0)
    {
#pragma omp critical
      ++numoutside;
      break;
    }
  }
}
