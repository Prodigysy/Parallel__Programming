#include <mpi.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#define ITERATIONS 10

using namespace std;
using namespace chrono;

class Matrix {
    unsigned int rows, cols;
    vector<vector<float>> elements;

public:
    Matrix();
    Matrix(unsigned int r, unsigned int c);
    Matrix(unsigned int r, unsigned int c, float minVal, float maxVal);
    Matrix operator*(const Matrix& other) const;

    const vector<vector<float>>& data() const { return elements; }
    vector<vector<float>>& data() { return elements; }

    friend ostream& operator<<(ostream& out, const Matrix& mat) {
        for (unsigned int i = 0; i < mat.rows; ++i) {
            for (unsigned int j = 0; j < mat.cols; ++j) {
                out << fixed << setprecision(4) << mat.elements[i][j] << ' ';
            }
            out << '\n';
        }
        return out;
    }
};

Matrix::Matrix() : rows(0), cols(0) {
    elements.resize(rows, vector<float>(cols, 0.0));
}

Matrix::Matrix(unsigned int r, unsigned int c) : rows(r), cols(c) {
    elements.resize(rows, vector<float>(cols, 0.0));
}

Matrix::Matrix(unsigned int r, unsigned int c, float minVal, float maxVal) : rows(r), cols(c) {
    elements.resize(rows, vector<float>(cols));
    srand(static_cast<unsigned>(time(nullptr)) + r + c);

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            elements[i][j] = minVal + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (maxVal - minVal));
        }
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    int rank, totalProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);

    int aRows, aCols, bRows, bCols;
    if (rank == 0) {
        aRows = rows;
        aCols = cols;
        bRows = other.rows;
        bCols = other.cols;
    }

    MPI_Bcast(&aRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&aCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (aCols != bRows) {
        if (rank == 0) cerr << "Incompatible matrix dimensions" << endl;
        return Matrix();
    }

    int base = aRows / totalProcs, remain = aRows % totalProcs;
    vector<int> sendCounts(totalProcs), displacements(totalProcs);
    for (int i = 0; i < totalProcs; ++i) {
        int slice = base + (i < remain ? 1 : 0);
        sendCounts[i] = slice * aCols;
        displacements[i] = (i == 0 ? 0 : displacements[i - 1] + sendCounts[i - 1]);
    }
    int localRows = sendCounts[rank] / aCols;

    vector<float> flatA;
    if (rank == 0) {
        flatA.resize(aRows * aCols);
        for (int i = 0; i < aRows; ++i)
            copy(elements[i].begin(), elements[i].end(), flatA.begin() + i * aCols);
    }

    vector<float> subA(localRows * aCols);
    MPI_Scatterv(flatA.data(), sendCounts.data(), displacements.data(), MPI_FLOAT,
                 subA.data(), sendCounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<float> flatB(bRows * bCols);
    if (rank == 0) {
        for (int i = 0; i < bRows; ++i)
            copy(other.elements[i].begin(), other.elements[i].end(), flatB.begin() + i * bCols);
    }
    MPI_Bcast(flatB.data(), bRows * bCols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<float> localResult(localRows * bCols, 0.0f);
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < bCols; ++j) {
            float val = 0.0f;
            for (int k = 0; k < aCols; ++k)
                val += subA[i * aCols + k] * flatB[k * bCols + j];
            localResult[i * bCols + j] = val;
        }
    }

    vector<int> recvCounts(totalProcs), recvDispls(totalProcs);
    for (int i = 0; i < totalProcs; ++i) {
        int r = sendCounts[i] / aCols;
        recvCounts[i] = r * bCols;
        recvDispls[i] = (i == 0 ? 0 : recvDispls[i - 1] + recvCounts[i - 1]);
    }

    vector<float> finalFlat;
    if (rank == 0) finalFlat.resize(aRows * bCols);

    MPI_Gatherv(localResult.data(), recvCounts[rank], MPI_FLOAT,
                finalFlat.data(), recvCounts.data(), recvDispls.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        Matrix output(aRows, bCols);
        for (int i = 0; i < aRows; ++i)
            copy(finalFlat.begin() + i * bCols, finalFlat.begin() + (i + 1) * bCols, output.elements[i].begin());
        return output;
    }
    return Matrix();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (unsigned int dim = 100; dim <= 1000; dim += 100) {
        double cumulative_time = 0.0;

        for (unsigned int test = 0; test < ITERATIONS; ++test) {
            int size = dim;
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            Matrix A(size, size), B(size, size);
            if (rank == 0) {
                A = Matrix(size, size, 1.0f, 99.0f);
                B = Matrix(size, size, 1.0f, 99.0f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto t_start = high_resolution_clock::now();
            Matrix C = A * B;
            auto t_end = high_resolution_clock::now();

            if (rank == 0) {
                cumulative_time += duration_cast<microseconds>(t_end - t_start).count() / 1e6;
            }
        }

        if (rank == 0) {
            double avg_time = cumulative_time / ITERATIONS;
            cout << dim << " " << avg_time << endl;
        }
    }

    MPI_Finalize();
    return 0;
}