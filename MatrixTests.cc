#include "Matrix.h"
#include <chrono>
#include <format>
#include <iostream>

#define COUNT_TIMES 10

using namespace std;
using namespace chrono;

int main() {
    for (unsigned int i = 100; i <= 1000; i += 100) {
        double total_duration = 0.0;
        cout << "Multiplying matrices of size " << i << "x" << i << "..." << endl;

        for (unsigned int j = 0; j < COUNT_TIMES; ++j) {
            Matrix a(i, i, 2.0, 100.0);
            Matrix b(i, i, 2.0, 100.0);

            string path = "C:/Users/user/Desktop/Parallel__Programming/results/";
            a.WriteData(format("{}a{}.txt", path, i));
            b.WriteData(format("{}b{}.txt", path, i));

            auto start_time = high_resolution_clock::now();
            Matrix result = a * b;
            auto end_time = high_resolution_clock::now();

            total_duration += duration_cast<microseconds>(end_time - start_time).count() / 1e6;

            result.WriteData(format("{}result{}.txt", path, i));
        }

        double avg_duration = total_duration / COUNT_TIMES;
        cout << "Average multiplication time: " << avg_duration << " seconds." << endl;
    }
    return 0;
}
