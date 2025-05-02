import numpy as np

def load_matrix_from_file(filepath):
    """Загружает матрицу из файла"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    rows, cols = map(int, lines[0].split())
    matrix_data = [] 
    for line in lines[1:rows+1]:  
        row = list(map(float, line.split()))
        matrix_data.append(row)
    
    return np.array(matrix_data)

def verify_matrix_multiplication(sizes, path_template):
    """Проверяет корректность умножения матриц"""
    for size in sizes:
        file_a = path_template.format(size=size, name='a')
        file_b = path_template.format(size=size, name='b')
        file_res = path_template.format(size=size, name='result')
        A = load_matrix_from_file(file_a)
        B = load_matrix_from_file(file_b)
        expected = load_matrix_from_file(file_res)

        computed = A @ B  # Альтернkатива np.dot
        
        if np.allclose(computed, expected, atol=1e-8, rtol=1e-5):
            print(f"Размер {size}x{size}: Верно ✓")
        else:
            print(f"Размер {size}x{size}: Неверно ✗")
            print(f"Максимальное расхождение: {np.max(np.abs(computed - expected))}")

def main():
    matrix_sizes = [100, 200, 300, 400, 500, 
                   600, 700, 800, 900, 1000]
    base_path = "C:/PP/ParallelProgramming/lab_1/tests/results/{name}{size}.txt"
    
    verify_matrix_multiplication(matrix_sizes, base_path)

if __name__ == "__main__":
    main()