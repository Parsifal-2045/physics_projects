// compile with mpic++ 
// run with mpirun -n numbers_of_processes -hostfile hosts ./a.out
// numbers_of_processes must be <= sums of the slots in the hostfile
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv)
{
    // init mpi using argc and argv
    MPI_Init(&argc, &argv);

    // MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Node name
    char name[MPI_MAX_PROCESSOR_NAME];
    int namLen;
    MPI_Get_processor_name(name, &namLen);

    std::cout << "Hello world! - From " << name << "'s process " << rank << " of " << size << '\n';
    MPI_Finalize();
}