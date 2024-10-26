#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

#include <mpi.h>

constexpr int nReps = 1000;

int main(int argc, char **argv)
{
    int nProcs, procId;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Barrier(MPI_COMM_WORLD);

    long long int N = 1LL << 22;
    std::vector<float> send(N);
    std::vector<float> receive(N);
    std::iota(send.begin(), send.end(), 0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i != nReps; ++i)
    {
        if (procId == 0)
        {
            MPI_Send(send.data(), N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(receive.data(), N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (procId == 1)
        {
            MPI_Recv(receive.data(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send.data(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;

    MPI_Finalize();

    if (procId == 0)
    {
        std::cout << "Elapsed ping-pong time: " << elapsed.count() << '\n';
    }
}