import numpy as np

def load_matrix(file_path):
    """Загружает матрицу из текстового файла в формате: первая строка — размеры, далее — значения."""
    with open(file_path, 'r') as file:
        header = file.readline().strip()
        row_count, col_count = map(int, header.split())
        values = [list(map(float, line.split())) for line in file]
    return np.array(values).reshape(row_count, col_count)

def verify_multiplication(matrix_a, matrix_b, expected_result, tolerance=1e-9):
    """Сравнивает произведение двух матриц с ожидаемым результатом."""
    computed = np.matmul(matrix_a, matrix_b)
    return np.allclose(computed, expected_result, atol=tolerance)

if __name__ == "__main__":
    base_path = "C:/Users/user/Desktop/Parallel__Programming/tests/results"
    dimensions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for dim in dimensions:
        path_a = f"{base_path}/a{dim}.txt"
        path_b = f"{base_path}/b{dim}.txt"
        path_result = f"{base_path}/result{dim}.txt"

        mat_a = load_matrix(path_a)
        mat_b = load_matrix(path_b)
        mat_res = load_matrix(path_result)

        if verify_multiplication(mat_a, mat_b, mat_res):
            print(f"✅ Size {dim}: Check successful.")
        else:
            print(f"❌ Size {dim}: Mismatch detected.")
