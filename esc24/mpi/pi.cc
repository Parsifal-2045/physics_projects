/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

This is the sequenctial version of the program.  It uses
the OpenMP timer.

History: Written by Tim Mattson, 11/99.

*/
#include <omp.h>
#include <mpi.h>
#include <chrono>

constexpr long num_steps = 1000000000;
constexpr double step = 1.0 / static_cast<double>(num_steps);

constexpr float percentage(float ratio)
{
   return ratio * 100;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int procId, nProcs;
   MPI_Comm_rank(MPI_COMM_WORLD, &procId);
   MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
   MPI_Barrier(MPI_COMM_WORLD);
   auto start = std::chrono::high_resolution_clock::now();

   int stepsPerProc = num_steps / nProcs;
   double sum = 0;
   double pi = 0;

   for (int i = procId * stepsPerProc; i != (procId + 1) * stepsPerProc; ++i)
   {
      double x = (i + 0.5) * step;
      sum += 4.0 / (1.0 + x * x);
   }

   MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   pi *= step;

   auto stop = std::chrono::high_resolution_clock::now();

   std::chrono::duration<float> elapsed = stop - start;
   float d = elapsed.count();
   std::cout << "Node " << procId << " out of " << nProcs << " computed "
             << percentage(1.f / nProcs) << "% of the total sum in " << d
             << " s\n";

   float cumulativeTime = 0;
   float maxTime = 0;
   float minTime = 0;
   MPI_Reduce(&d,              // node local variable
              &cumulativeTime, // global variable where reduction happens
              1,               // length of the array
              MPI_FLOAT,       // MPY_TYPE of the data
              MPI_SUM,         // MPI operation
              0,               // starting value
              MPI_COMM_WORLD);
   MPI_Reduce(&d, &maxTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&d, &minTime, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
   float meanTime = cumulativeTime / static_cast<float>(nProcs);

   MPI_Finalize();

   if (procId == 0)
   {
      std::cout << "Pi with " << num_steps << " steps is " << pi << '\n'
                << "Node time: \n"
                << "\t Min: " << minTime << '\n'
                << "\t Max: " << maxTime << '\n'
                << "\t Mean: " << meanTime << '\n';
   }
}
