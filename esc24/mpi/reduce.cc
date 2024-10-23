#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int nNodes, procId;
    MPI_Comm_size(MPI_COMM_WORLD, &nNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    int sum = 0;
    int buf = procId;

    MPI_Reduce(&buf, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    if (procId == 0)
    {
        std::cout << "Sum of all procIds: " << sum << '\n';
    }
}