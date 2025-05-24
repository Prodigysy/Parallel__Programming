import numpy as np

def load_matrix(file_path):
    """Загружает матрицу из текстового файла."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dims = list(map(int, lines[0].strip().split()))
    values = [list(map(float, row.strip().split())) for row in lines[1:]]

    return np.array(values).reshape(dims)

def verify_matrix_multiplication(size_list, base_path):
    """Проверяет корректность результатов перемножения матриц."""
    for n in size_list:
        path_a = f"{base_path}/matrix_A_{n}.txt"
        path_b = f"{base_path}/matrix_B_{n}.txt"
        path_c = f"{base_path}/matrix_C_{n}.txt"

        mat_a = load_matrix(path_a)
        mat_b = load_matrix(path_b)
        mat_c = load_matrix(path_c)

        expected = np.matmul(mat_a, mat_b)

        if np.allclose(expected, mat_c, atol=1e-9):
            print(f"[{n}x{n}] ✔ Проверка пройдена")
        else:
            print(f"[{n}x{n}] ❌ Ошибка в результате")

if __name__ == "__main__":
    test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results_directory = "C:/Users/user/Desktop/Parallel__Programming/results"
    verify_matrix_multiplication(test_sizes, results_directory)
