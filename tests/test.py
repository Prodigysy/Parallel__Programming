import numpy as np
from pathlib import Path

def load_matrix_data(file_path):
    """Загружает матрицу из текстового файла"""
    with Path(file_path).open() as file:
        content = file.read().splitlines()

    dimensions = content[0].strip().split()
    n_rows, n_cols = int(dimensions[0]), int(dimensions[1])
    matrix_values = []
    for row in content[1:n_rows+1]:
        values = [float(x) for x in row.strip().split()]
        matrix_values.append(values)
    
    return np.array(matrix_values, dtype=np.float64)

def verify_matrix_multiplication(test_sizes, base_dir):
    """Проверяет корректность умножения матриц для разных размеров"""
    for size in test_sizes:
        matrix_a_path = base_dir / f"a{size}.txt"
        matrix_b_path = base_dir / f"b{size}.txt"
        expected_result_path = base_dir / f"result{size}.txt"
        
        try:
            mat_a = load_matrix_data(matrix_a_path)
            mat_b = load_matrix_data(matrix_b_path)
            expected = load_matrix_data(expected_result_path)

            computed = mat_a @ mat_b  

            if np.allclose(computed, expected, atol=1e-8, rtol=1e-5):
                print(f"Matrix {size}x{size}: Verification successful [✓]")
            else:
                max_diff = np.max(np.abs(computed - expected))
                print(f"Matrix {size}x{size}: Verification failed [✗] (Max diff: {max_diff:.2e})")
                
        except FileNotFoundError as e:
            print(f"Error processing size {size}: {str(e)}")
            continue

if __name__ == "__main__":
    test_dimensions = [100, 200, 300, 400, 500, 
                      600, 700, 800, 900, 1000]
    base_directory = Path("C:/Users/user/Desktop/PP/Parallel__Programming/tests/results")
    
    verify_matrix_multiplication(test_dimensions, base_directory)