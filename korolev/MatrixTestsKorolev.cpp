#include <mpi.h>
#include <chrono>
#include <format>
#include <iostream>

#define COUNT_TIMES 10

using namespace std;
using namespace chrono;

class Matrix {
    unsigned int _rows;
    unsigned int _cols;
    vector<vector<float> > _data;
public:
    Matrix();
    Matrix(unsigned int rows, unsigned int cols);
    Matrix(unsigned int rows, unsigned int cols, const float lower, const float upper);
    Matrix operator*(const Matrix& other) const;

    const vector<vector<float> >& Data() const { return _data; }
    vector<vector<float> >& Data() { return _data; }

    friend ostream& operator<<(ostream& os, const Matrix& matrix) {
        for (unsigned int i = 0; i < matrix._rows; i++) {
            for (unsigned int j = 0; j < matrix._cols; j++) {
                os << fixed << setprecision(4) << matrix._data[i][j] << ' ';
            }
            os << '\n';
        }
        return os;
    }

};

Matrix::Matrix() : _rows(0), _cols(0) {
    _data.resize(_rows, vector<float>(_cols, 0.0));
}

Matrix::Matrix(unsigned int rows, unsigned int cols) : _rows(rows), _cols(cols) {
    _data.resize(_rows, vector<float>(_cols, 0.0));
}

Matrix::Matrix(unsigned int rows, unsigned int cols, const float lower, const float upper) {
    _rows = rows;
    _cols = cols;
    _data.resize(_rows, vector<float>(_cols));

    srand(static_cast<unsigned>(time(nullptr)));

    for (unsigned int i = 0; i < _rows; ++i) {
        for (unsigned int j = 0; j < _cols; ++j) {
            _data[i][j] = lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (upper - lower));
        }
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsA, colsA, rowsB, colsB;
    if (rank == 0) {
        rowsA = _rows;
        colsA = _cols;
        rowsB = other._rows;
        colsB = other._cols;
    }
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (colsA != rowsB) {
        if (rank == 0) cerr << "Dimension mismatch for multiplication" << endl;
        return Matrix();
    }

    int baseRows = rowsA / size;
    int rem = rowsA % size;
    vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int chunk = baseRows + (i < rem ? 1 : 0);
        counts[i] = chunk * colsA;
        displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }
    int localRows = counts[rank] / colsA;

    vector<float> flatA;
    if (rank == 0) {
        flatA.resize(rowsA * colsA);
        for (int i = 0; i < rowsA; ++i)
            copy(_data[i].begin(), _data[i].end(), flatA.begin() + i * colsA);
    }
    vector<float> localA(localRows * colsA);
    MPI_Scatterv(flatA.data(), counts.data(), displs.data(), MPI_FLOAT,
        localA.data(), counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<float> flatB(rowsB * colsB);
    if (rank == 0) {
        for (int i = 0; i < rowsB; ++i)
            copy(other._data[i].begin(), other._data[i].end(), flatB.begin() + i * colsB);
    }
    MPI_Bcast(flatB.data(), rowsB * colsB, MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<float> localC(localRows * colsB, 0.0f);
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k)
                sum += localA[i * colsA + k] * flatB[k * colsB + j];
            localC[i * colsB + j] = sum;
        }
    }

    vector<int> recvCounts(size), recvDispls(size);
    for (int i = 0; i < size; ++i) {
        int rows = counts[i] / colsA;
        recvCounts[i] = rows * colsB;
        recvDispls[i] = (i == 0 ? 0 : recvDispls[i - 1] + recvCounts[i - 1]);
    }
    vector<float> flatResult;
    if (rank == 0) flatResult.resize(rowsA * colsB);
    MPI_Gatherv(localC.data(), recvCounts[rank], MPI_FLOAT,
        flatResult.data(), recvCounts.data(), recvDispls.data(), MPI_FLOAT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        Matrix result(rowsA, colsB);
        for (int i = 0; i < rowsA; ++i)
            copy(flatResult.begin() + i * colsB,
                flatResult.begin() + (i + 1) * colsB,
                result._data[i].begin());
        return result;
    }
    return Matrix();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (unsigned int i = 100; i <= 1000; i += 100) {
        double total_duration = 0.0;

        for (unsigned int j = 0; j < COUNT_TIMES; ++j) {
            int matrix_size = i;
            MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            Matrix a(matrix_size, matrix_size);
            Matrix b(matrix_size, matrix_size);

            if (rank == 0) {
                a = Matrix(matrix_size, matrix_size, 2.0f, 100.0f);
                b = Matrix(matrix_size, matrix_size, 2.0f, 100.0f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto start_time = high_resolution_clock::now();
            Matrix result = a * b;
            auto end_time = high_resolution_clock::now();

            if (rank == 0) {
                total_duration += duration_cast<microseconds>(end_time - start_time).count() / 1e6;
            }
        }

        if (rank == 0) {
            double avg_duration = total_duration / COUNT_TIMES;
            cout << i << " " << avg_duration << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

