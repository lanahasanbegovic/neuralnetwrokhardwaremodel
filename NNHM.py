import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from collections import deque
import pandas as pd


@dataclass
class ProcessingElement:
    """
    Represents a single processing element in the systolic array.
    Each PE can store a weight, accumulate partial sums, and pass data.
    """
    row: int
    col: int
    weight: float = 0.0
    accumulator: float = 0.0
    input_buffer: Optional[float] = None
    output_buffer: Optional[float] = None
    partial_sum_buffer: Optional[float] = None

    def reset(self):
        """Reset the PE state for new computation"""
        self.accumulator = 0.0
        self.input_buffer = None
        self.output_buffer = None
        self.partial_sum_buffer = None

    def compute(self, input_val: Optional[float], partial_sum: Optional[float]) -> Tuple[
        Optional[float], Optional[float]]:
        """
        Perform multiply-accumulate operation and pass values
        Returns: (output_to_next_pe, updated_partial_sum)
        """
        # Store input for next cycle
        self.input_buffer = input_val

        # Perform MAC operation if we have valid input
        if input_val is not None and partial_sum is not None:
            mac_result = input_val * self.weight + partial_sum
            self.accumulator = mac_result
            return input_val, mac_result
        elif input_val is not None and partial_sum is None:
            mac_result = input_val * self.weight
            self.accumulator = mac_result
            return input_val, mac_result

        return input_val, partial_sum


class SystolicArray:
    """
    Simulates a systolic array hardware accelerator for matrix multiplication.
    Implements the classic weight-stationary dataflow.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.pe_grid = [[ProcessingElement(i, j) for j in range(cols)] for i in range(rows)]
        self.cycle_count = 0
        self.computation_history = []
        self.data_flow_history = []

    def load_weights(self, weight_matrix: np.ndarray):
        """Load weights into the processing elements"""
        weight_rows, weight_cols = weight_matrix.shape

        # Clear all weights first
        for i in range(self.rows):
            for j in range(self.cols):
                self.pe_grid[i][j].weight = 0.0

        # Load weights up to the array size
        for i in range(min(self.rows, weight_rows)):
            for j in range(min(self.cols, weight_cols)):
                self.pe_grid[i][j].weight = weight_matrix[i, j]

    def reset_array(self):
        """Reset all processing elements"""
        self.cycle_count = 0
        self.computation_history = []
        self.data_flow_history = []
        for i in range(self.rows):
            for j in range(self.cols):
                self.pe_grid[i][j].reset()

    def matrix_multiply(self, input_matrix: np.ndarray, weight_matrix: np.ndarray, verbose: bool = False) -> Tuple[
        np.ndarray, int]:
        """
        Perform matrix multiplication using the systolic array
        Uses output-stationary dataflow for simplicity
        Returns: (result_matrix, total_cycles)
        """
        if input_matrix.shape[1] != weight_matrix.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")

        # Check if matrices fit in the array
        input_rows, input_cols = input_matrix.shape
        weight_rows, weight_cols = weight_matrix.shape

        if weight_rows > self.rows or weight_cols > self.cols:
            if verbose:
                print(f"Warning: Weight matrix {weight_matrix.shape} larger than array {(self.rows, self.cols)}")

        # Load weights
        self.load_weights(weight_matrix)
        self.reset_array()

        # Result matrix
        result = np.zeros((input_rows, weight_cols))

        # For simplicity, let's use a more straightforward approach
        # Process one output element at a time using the systolic principle

        total_cycles = 0

        if verbose:
            print(f"Starting systolic array computation...")
            print(f"Input matrix: {input_matrix.shape}, Weight matrix: {weight_matrix.shape}")
            print(f"Systolic array: {self.rows}x{self.cols}")

        # For each output position
        for out_row in range(input_rows):
            for out_col in range(min(weight_cols, self.cols)):
                # Reset accumulators for this output element
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.pe_grid[i][j].reset()

                # Process the dot product for this output element
                cycles_for_element = 0

                # Simulate the systolic computation for this element
                for k in range(min(input_cols, self.rows)):
                    cycle_state = {'cycle': total_cycles + cycles_for_element, 'pe_states': []}

                    # In cycle k, process input_matrix[out_row, k] * weight_matrix[k, out_col]
                    if k < weight_rows and out_col < weight_cols:
                        pe = self.pe_grid[k][out_col]
                        input_val = input_matrix[out_row, k]

                        # Perform MAC operation
                        if k == 0:
                            partial_sum = 0.0
                        else:
                            partial_sum = pe.accumulator

                        _, updated_sum = pe.compute(input_val, partial_sum)

                        # Record state
                        cycle_state['pe_states'].append({
                            'row': k, 'col': out_col,
                            'input': input_val,
                            'weight': pe.weight,
                            'partial_sum_in': partial_sum,
                            'partial_sum_out': updated_sum,
                            'accumulator': pe.accumulator
                        })

                    cycles_for_element += 1
                    self.computation_history.append(cycle_state)

                # Store the final result
                if out_col < weight_cols and input_cols > 0:
                    final_pe = self.pe_grid[min(input_cols - 1, self.rows - 1)][out_col]
                    result[out_row, out_col] = final_pe.accumulator

                total_cycles += cycles_for_element

                if verbose and out_row < 2 and out_col < 2:
                    print(
                        f"Output[{out_row},{out_col}] = {result[out_row, out_col]:.3f} (took {cycles_for_element} cycles)")

        self.cycle_count = total_cycles
        return result, total_cycles


class CPUSystolicComparison:
    """
    Compare CPU-based matrix multiplication with systolic array simulation
    """

    def __init__(self):
        self.cpu_times = []
        self.systolic_cycles = []
        self.matrix_sizes = []

    def cpu_matrix_multiply(self, A: np.ndarray, B: np.ndarray, method: str = 'naive') -> Tuple[np.ndarray, float, int]:
        """
        CPU-based matrix multiplication with operation counting
        """
        if method == 'naive':
            return self._naive_multiply(A, B)
        elif method == 'numpy':
            return self._numpy_multiply(A, B)
        else:
            raise ValueError("Method must be 'naive' or 'numpy'")

    def _naive_multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Naive triple-loop matrix multiplication"""
        start_time = time.time()
        m, n = A.shape
        n2, p = B.shape

        if n != n2:
            raise ValueError("Matrix dimensions don't match")

        C = np.zeros((m, p))
        operations = 0

        for i in range(m):
            for j in range(p):
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]
                    operations += 2  # One multiply, one add

        end_time = time.time()
        return C, end_time - start_time, operations

    def _numpy_multiply(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """NumPy optimized matrix multiplication"""
        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()

        # Estimate operations (theoretical)
        m, n = A.shape
        p = B.shape[1]
        operations = m * n * p * 2  # Each element requires n multiplies and n-1 adds

        return C, end_time - start_time, operations

    def benchmark_comparison(self, sizes: List[int], array_size: Tuple[int, int] = (8, 8)) -> pd.DataFrame:
        """
        Benchmark CPU vs Systolic Array performance
        """
        results = []

        for size in sizes:
            print(f"Benchmarking matrix size: {size}x{size}")

            # Generate random matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # CPU Naive
            C_cpu_naive, time_cpu_naive, ops_cpu_naive = self.cpu_matrix_multiply(A, B, 'naive')

            # CPU NumPy
            C_cpu_numpy, time_cpu_numpy, ops_cpu_numpy = self.cpu_matrix_multiply(A, B, 'numpy')

            # Systolic Array (if matrices fit)
            max_dim = max(size, size)  # For square matrices
            if max_dim <= 10:  # Reasonable limit for systolic array simulation
                systolic = SystolicArray(max(8, size), max(8, size))  # Ensure minimum size
                try:
                    C_systolic, cycles_systolic = systolic.matrix_multiply(A, B)

                    # Extract the relevant portion for comparison
                    C_systolic_trimmed = C_systolic[:size, :size]

                    # Verify correctness
                    error = np.max(np.abs(C_cpu_numpy - C_systolic_trimmed))

                    results.append({
                        'matrix_size': size,
                        'cpu_naive_time': time_cpu_naive,
                        'cpu_numpy_time': time_cpu_numpy,
                        'systolic_cycles': cycles_systolic,
                        'cpu_naive_ops': ops_cpu_naive,
                        'cpu_numpy_ops': ops_cpu_numpy,
                        'systolic_error': error,
                        'systolic_throughput': ops_cpu_numpy / cycles_systolic if cycles_systolic > 0 else 0
                    })
                except Exception as e:
                    print(f"Systolic array failed for size {size}: {e}")
                    results.append({
                        'matrix_size': size,
                        'cpu_naive_time': time_cpu_naive,
                        'cpu_numpy_time': time_cpu_numpy,
                        'systolic_cycles': None,
                        'cpu_naive_ops': ops_cpu_naive,
                        'cpu_numpy_ops': ops_cpu_numpy,
                        'systolic_error': None,
                        'systolic_throughput': None
                    })
            else:
                results.append({
                    'matrix_size': size,
                    'cpu_naive_time': time_cpu_naive,
                    'cpu_numpy_time': time_cpu_numpy,
                    'systolic_cycles': None,
                    'cpu_naive_ops': ops_cpu_naive,
                    'cpu_numpy_ops': ops_cpu_numpy,
                    'systolic_error': None,
                    'systolic_throughput': None
                })

        return pd.DataFrame(results)


def visualize_systolic_array_operation(systolic: SystolicArray, cycle_range: Tuple[int, int] = (0, 5)):
    """
    Visualize the systolic array operation over multiple cycles
    """
    if not systolic.computation_history:
        print("No computation history available. Run matrix multiplication first.")
        return

    start_cycle, end_cycle = cycle_range
    end_cycle = min(end_cycle, len(systolic.computation_history))

    if end_cycle <= start_cycle:
        print("No cycles to display")
        return

    num_plots = min(5, end_cycle - start_cycle)  # Limit to 5 plots max
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    for idx, cycle in enumerate(range(start_cycle, start_cycle + num_plots)):
        if cycle >= len(systolic.computation_history):
            break

        ax = axes[idx]
        cycle_data = systolic.computation_history[cycle]

        # Create heatmap data - initialize with zeros
        pe_values = np.zeros((systolic.rows, systolic.cols))
        annot_matrix = np.full((systolic.rows, systolic.cols), "", dtype=object)

        # Fill in data from active PEs only
        for pe_state in cycle_data['pe_states']:
            row, col = pe_state['row'], pe_state['col']
            if row < systolic.rows and col < systolic.cols:
                pe_values[row, col] = pe_state['accumulator'] if pe_state['accumulator'] else 0

                # Create annotation
                input_val = pe_state['input'] if pe_state['input'] is not None else 0
                weight = pe_state['weight']
                acc = pe_state['accumulator'] if pe_state['accumulator'] else 0

                annot_matrix[row, col] = f"I:{input_val:.1f}\nW:{weight:.1f}\nA:{acc:.1f}"

        # Fill empty cells with default annotation
        for i in range(systolic.rows):
            for j in range(systolic.cols):
                if annot_matrix[i, j] == "":
                    weight = systolic.pe_grid[i][j].weight
                    annot_matrix[i, j] = f"I:0.0\nW:{weight:.1f}\nA:0.0"

        # Plot heatmap
        sns.heatmap(pe_values, annot=annot_matrix, fmt='', ax=ax, cmap='viridis',
                    cbar=False, square=True, linewidths=1, annot_kws={'size': 8})
        ax.set_title(f'Cycle {cycle}')
        ax.set_xlabel('PE Column')
        ax.set_ylabel('PE Row')

    plt.tight_layout()
    plt.show()


def plot_performance_comparison(benchmark_df: pd.DataFrame):
    """
    Plot performance comparison between CPU and Systolic Array
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Execution time comparison
    valid_data = benchmark_df.dropna(subset=['systolic_cycles'])

    ax1.plot(benchmark_df['matrix_size'], benchmark_df['cpu_naive_time'], 'o-', label='CPU Naive', linewidth=2)
    ax1.plot(benchmark_df['matrix_size'], benchmark_df['cpu_numpy_time'], 's-', label='CPU NumPy', linewidth=2)
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Cycles vs Operations
    if not valid_data.empty:
        ax2.plot(valid_data['matrix_size'], valid_data['systolic_cycles'], 'o-', label='Systolic Cycles', color='red',
                 linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(valid_data['matrix_size'], valid_data['cpu_numpy_ops'], 's-', label='CPU Operations',
                      color='blue', linewidth=2)
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Systolic Cycles', color='red')
        ax2_twin.set_ylabel('CPU Operations', color='blue')
        ax2.set_title('Cycles vs Operations')
        ax2.grid(True, alpha=0.3)

    # Throughput comparison
    if not valid_data.empty and 'systolic_throughput' in valid_data.columns:
        cpu_throughput = valid_data['cpu_numpy_ops'] / valid_data['cpu_numpy_time']
        ax3.plot(valid_data['matrix_size'], cpu_throughput, 'o-', label='CPU Throughput (ops/sec)', linewidth=2)
        ax3.plot(valid_data['matrix_size'], valid_data['systolic_throughput'], 's-',
                 label='Systolic Throughput (ops/cycle)', linewidth=2)
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Throughput')
        ax3.set_title('Throughput Comparison')
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

    # Error analysis
    if not valid_data.empty:
        ax4.plot(valid_data['matrix_size'], valid_data['systolic_error'], 'o-', color='green', linewidth=2)
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Maximum Error')
        ax4.set_title('Systolic Array Numerical Error')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_user_matrix_size():
    """
    Get matrix size from user input with validation
    """
    while True:
        try:
            size_input = input("Enter matrix size (n) for n x n matrices [2-12]: ").strip()
            size = int(size_input)
            
            if size < 2:
                print("Matrix size must be at least 2. Please try again.")
                continue
            elif size > 12:
                print("Matrix size limited to 12 for visualization purposes. Please try again.")
                continue
            
            return size
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def get_user_matrix_input(size: int, matrix_name: str):
    """
    Get matrix values from user input or use random generation
    """
    print(f"\nFor {matrix_name} ({size}x{size}):")
    print("1. Enter values manually")
    print("2. Generate random values")
    print("3. Use identity matrix")
    print("4. Use ones matrix")
    
    while True:
        try:
            choice = input("Select option [1-4]: ").strip()
            
            if choice == '1':
                # Manual input
                matrix = np.zeros((size, size), dtype=np.float32)
                print(f"Enter values for {matrix_name} row by row (space-separated):")
                for i in range(size):
                    while True:
                        try:
                            row_input = input(f"Row {i+1}: ").strip()
                            values = list(map(float, row_input.split()))
                            if len(values) != size:
                                print(f"Please enter exactly {size} values.")
                                continue
                            matrix[i] = values
                            break
                        except ValueError:
                            print("Please enter valid numbers separated by spaces.")
                return matrix
            
            elif choice == '2':
                # Random values
                matrix = np.random.randn(size, size).astype(np.float32)
                print(f"Generated random {matrix_name}:")
                print(matrix)
                return matrix
            
            elif choice == '3':
                # Identity matrix
                matrix = np.eye(size, dtype=np.float32)
                print(f"Generated identity {matrix_name}:")
                print(matrix)
                return matrix
            
            elif choice == '4':
                # Ones matrix
                matrix = np.ones((size, size), dtype=np.float32)
                print(f"Generated ones {matrix_name}:")
                print(matrix)
                return matrix
            
            else:
                print("Please select a valid option (1-4).")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


# Updated Demonstration and Testing
def main_demonstration():
    """
    Main demonstration of the neural network accelerator model with user input
    """
    print("=" * 60)
    print("NEURAL NETWORK ACCELERATOR HARDWARE MODEL")
    print("=" * 60)

    # 1. Get matrix size from user
    print("\n1. MATRIX SETUP")
    print("-" * 40)
    
    matrix_size = get_user_matrix_size()
    if matrix_size is None:
        return None, None
    
    print(f"\nCreating {matrix_size}x{matrix_size} matrices...")
    
    # 2. Get matrix inputs from user
    A = get_user_matrix_input(matrix_size, "Input Matrix A")
    if A is None:
        return None, None
    
    B = get_user_matrix_input(matrix_size, "Weight Matrix B")
    if B is None:
        return None, None

    print("\n2. BASIC SYSTOLIC ARRAY DEMONSTRATION")
    print("-" * 40)

    print("Input Matrix A:")
    print(A)
    print("\nWeight Matrix B:")
    print(B)

    # Expected result using NumPy
    expected = np.dot(A, B)
    print("\nExpected Result (NumPy):")
    print(expected)

    # 3. Test systolic array
    array_size = max(8, matrix_size)  # Ensure minimum array size
    systolic = SystolicArray(array_size, array_size)
    result_systolic, cycles = systolic.matrix_multiply(A, B, verbose=True)

    print(f"\nSystolic Array Result:")
    print(result_systolic[:matrix_size, :matrix_size])  # Show only relevant portion
    print(f"Completed in {cycles} cycles")
    
    # Calculate error
    error = np.max(np.abs(expected - result_systolic[:matrix_size, :matrix_size]))
    print(f"Maximum Error: {error:.10f}")

    # 4. Visualize systolic array operation
    print("\n3. SYSTOLIC ARRAY VISUALIZATION")
    print("-" * 40)
    
    # Ask user if they want to see visualization
    while True:
        try:
            viz_choice = input("Show cycle-by-cycle visualization? [y/n]: ").strip().lower()
            if viz_choice in ['y', 'yes']:
                visualize_systolic_array_operation(systolic, (0, min(5, len(systolic.computation_history))))
                break
            elif viz_choice in ['n', 'no']:
                print("Skipping visualization.")
                break
            else:
                print("Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\nSkipping visualization.")
            break

    # 5. Performance benchmark with user's size
    print("\n4. PERFORMANCE BENCHMARK")
    print("-" * 40)

    # Ask user if they want to run benchmark
    while True:
        try:
            bench_choice = input("Run performance benchmark? [y/n]: ").strip().lower()
            if bench_choice in ['y', 'yes']:
                comparator = CPUSystolicComparison()
                sizes = list(range(2, min(matrix_size + 3, 9)))  # Include user's size and some others
                if matrix_size not in sizes:
                    sizes.append(matrix_size)
                sizes.sort()
                
                benchmark_results = comparator.benchmark_comparison(sizes, array_size=(array_size, array_size))
                
                print("\nBenchmark Results:")
                print(benchmark_results.to_string(index=False))
                
                # Plot performance comparison
                print("\n5. PERFORMANCE VISUALIZATION")
                print("-" * 40)
                plot_performance_comparison(benchmark_results)
                break
                
            elif bench_choice in ['n', 'no']:
                print("Skipping benchmark.")
                benchmark_results = None
                break
            else:
                print("Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\nSkipping benchmark.")
            benchmark_results = None
            break

    # 6. Analysis and summary
    print("\n6. ANALYSIS SUMMARY")
    print("-" * 40)

    print(f"\nMatrix Size: {matrix_size}x{matrix_size}")
    print(f"Systolic Array Size: {array_size}x{array_size}")
    print(f"Total Cycles: {cycles}")
    print(f"Numerical Accuracy: {error:.2e}")
    
    # Theoretical analysis
    theoretical_ops = matrix_size * matrix_size * matrix_size * 2  # MAC operations
    print(f"Theoretical Operations: {theoretical_ops}")
    if cycles > 0:
        print(f"Operations per Cycle: {theoretical_ops / cycles:.2f}")

    print("\nSystolic Array Advantages:")
    print("• High throughput for matrix operations")
    print("• Predictable, regular data flow")
    print("• Good for pipeline parallelism")
    print("• Energy efficient for neural network inference")

    print("\nSystolic Array Limitations:")
    print("• Fixed array size limits matrix dimensions")
    print("• Setup overhead for small matrices")
    print("• Less flexible than general-purpose CPUs")
    print("• Memory bandwidth requirements")

    return benchmark_results, systolic


if __name__ == "__main__":
    # Run the demonstration
    benchmark_data, systolic_array = main_demonstration()
    
    if benchmark_data is not None:
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. Systolic arrays provide regular, predictable computation patterns")
        print("2. They excel at matrix operations with high data reuse")
        print("3. Performance benefits become apparent at larger scales")
        print("4. Trade-offs exist between flexibility and efficiency")
    else:
        print("\nDemonstration ended by user.")