#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "cuda_mul.h"

using namespace std;

class Matrix {
	unsigned int _rows;
	unsigned int _cols;
	vector<vector<float>> _data;
public:
	Matrix();
	Matrix(unsigned int rows, unsigned int cols);
	Matrix(unsigned int rows, unsigned int cols, const float lower, const float upper);
	Matrix operator*(const Matrix& other) const;

	const vector<vector<float>>& Data() const { return _data; }
	vector<vector<float>>& Data() { return _data; }

	void ReadData(const string& filename);
	void WriteData(const string& filename) const;

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
	_data.resize(_rows, std::vector<float>(_cols, 0.0));
}

Matrix::Matrix(unsigned int rows, unsigned int cols) : _rows(rows), _cols(cols) {
	_data.resize(_rows, std::vector<float>(_cols, 0.0));
}

Matrix::Matrix(unsigned int rows, unsigned int cols, const float lower, const float upper) {
	_rows = rows;
	_cols = cols;
	_data.resize(_rows, std::vector<float>(_cols));

	srand(static_cast<unsigned>(time(nullptr)));

	for (unsigned int i = 0; i < _rows; ++i) {
		for (unsigned int j = 0; j < _cols; ++j) {
			_data[i][j] = lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (upper - lower));
		}
	}
}

void Matrix::ReadData(const string& filename) {
	ifstream fin(filename);
	if (!fin) {
		cerr << "Error during opening file in Read: " << filename << endl;
		return;
	}

	if (!(fin >> _rows >> _cols)) {
		cerr << "Error reading matrix dimensions from file: " << filename << endl;
		return;
	}

	_data.assign(_rows, std::vector<float>(_cols));

	for (unsigned int i = 0; i < _rows; i++) {
		for (unsigned int j = 0; j < _cols; j++)
		{
			fin >> _data[i][j];
		}
	}
}
void Matrix::WriteData(const string& filename) const {
	ofstream fout(filename);
	if (!fout) {
		cerr << "Error during opening file in Write: " << filename << endl;
		return;
	}
	fout << _rows << " " << _cols << std::endl;
	for (unsigned int i = 0; i < _rows; i++) {
		for (unsigned int j = 0; j < _cols; j++) {
			fout << fixed << setprecision(4) << _data[i][j] << ' ';
		}
		fout << '\n';
	}
}
Matrix Matrix::operator*(const Matrix& other) const {
	int rowsA = this->_data.size();
	int colsA = this->_data[0].size();
	int rowsB = other._data.size();
	int colsB = other._data[0].size();

	if (colsA != rowsB) {
		std::cerr << "Matrix dimensions do not match for multiplication." << std::endl;
		return Matrix();
	}

	std::vector<float> flatA(rowsA * colsA);
	std::vector<float> flatB(rowsB * colsB);
	std::vector<float> flatC(rowsA * colsB, 0.0f);

	for (int i = 0; i < rowsA; ++i)
		std::copy(_data[i].begin(), _data[i].end(), flatA.begin() + i * colsA);
	for (int i = 0; i < rowsB; ++i)
		std::copy(other._data[i].begin(), other._data[i].end(), flatB.begin() + i * colsB);

	cudaMatrixMultiply(flatA, flatB, flatC, rowsA, colsA, colsB);

	Matrix result(rowsA, colsB);
	for (int i = 0; i < rowsA; ++i)
		std::copy(flatC.begin() + i * colsB, flatC.begin() + (i + 1) * colsB, result._data[i].begin());

	return result;
}