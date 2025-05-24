#include <matrix/matrix.h>
#include <chrono>
#include <format>
#include <iostream>
#include <mpi.h>

#define COUNT_TIMES 10

using namespace std;
using namespace chrono;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (unsigned int i = 100; i <= 1000; i += 100) {
        double total_duration = 0.0;

        if (rank == 0) {
            cout << "Multiplying matrices of size " << i << "x" << i << "..." << endl;
        }

        for (unsigned int j = 0; j < COUNT_TIMES; ++j) {
            int matrix_size = i;
            MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            Matrix a(matrix_size, matrix_size);
            Matrix b(matrix_size, matrix_size);

            if (rank == 0) {
                a = Matrix(matrix_size, matrix_size, 2.0f, 100.0f);
                b = Matrix(matrix_size, matrix_size, 2.0f, 100.0f);

                if (j == 0) {
                    string path = "C:/Users/user/Desktop/Parallel__Programming/tests/results/";
                    a.WriteData(format("{}a{}.txt", path, i));
                    b.WriteData(format("{}b{}.txt", path, i));
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto start_time = high_resolution_clock::now();
            Matrix result = a * b;
            auto end_time = high_resolution_clock::now();

            if (rank == 0) {
                total_duration += duration_cast<microseconds>(end_time - start_time).count() / 1e6;

                if (j == 0) {
                    string path = "C:/Users/user/Desktop/Parallel__Programming/tests/results/";
                    result.WriteData(format("{}result{}.txt", path, i));
                }
            }
        }

        if (rank == 0) {
            double avg_duration = total_duration / COUNT_TIMES;
            cout << "Average multiplication time: " << avg_duration << " seconds." << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

