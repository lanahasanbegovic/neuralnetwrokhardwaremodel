import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import queue
import time
from testing import (
    SystolicArray, 
    CPUSystolicComparison, 
    visualize_systolic_array_operation, 
    plot_performance_comparison
)
import tkinter.font as tkFont
import matplotlib.pyplot as plt

# Set matplotlib font
plt.rcParams['font.family'] = 'Space Mono'




class SystolicArrayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Accelerator - Systolic Array Simulator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables FIRST
        self.matrix_a = None
        self.matrix_b = None
        self.result_matrix = None
        self.systolic_array = None
        self.benchmark_results = None
        self.current_size = 3
        
        # Initialize GUI variables early
        self.size_var = tk.StringVar(value="3")
        self.array_size_var = tk.StringVar(value="8")
        
        # Create queue for thread communication
        self.result_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize with default matrices
        self.generate_random_matrices()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame with notebook for tabs
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel at the top FIRST
        self.setup_control_panel(main_frame)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Setup tabs
        self.setup_matrix_tab()
        self.setup_computation_tab()
        self.setup_visualization_tab()
        self.setup_benchmark_tab()
        
    def setup_control_panel(self, parent):
        """Setup the main control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Matrix size control
        ttk.Label(control_frame, text="Matrix Size:").grid(row=0, column=0, padx=(0, 5))
        size_spinbox = ttk.Spinbox(control_frame, from_=2, to=10, width=5, textvariable=self.size_var)
        size_spinbox.grid(row=0, column=1, padx=(0, 10))
        size_spinbox.bind('<Return>', self.on_size_change)
        
        # Array size control
        ttk.Label(control_frame, text="Array Size:").grid(row=0, column=2, padx=(10, 5))
        array_spinbox = ttk.Spinbox(control_frame, from_=4, to=16, width=5, textvariable=self.array_size_var)
        array_spinbox.grid(row=0, column=3, padx=(0, 10))
        
        # Matrix generation buttons
        ttk.Button(control_frame, text="Random Matrices", 
                  command=self.generate_random_matrices).grid(row=0, column=4, padx=(10, 5))
        ttk.Button(control_frame, text="Identity Matrices", 
                  command=self.generate_identity_matrices).grid(row=0, column=5, padx=(0, 5))
        ttk.Button(control_frame, text="Ones Matrices", 
                  command=self.generate_ones_matrices).grid(row=0, column=6, padx=(0, 10))
        
        # Main action buttons
        ttk.Button(control_frame, text="Run Computation", 
                  command=self.run_computation, style="Accent.TButton").grid(row=0, column=7, padx=(10, 5))
        ttk.Button(control_frame, text="Run Benchmark", 
                  command=self.run_benchmark).grid(row=0, column=8, padx=(0, 5))
        
    def setup_matrix_tab(self):
        """Setup the matrix input/display tab"""
        self.matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_frame, text="Matrix Setup")
        
        # Create three columns for the matrices
        left_frame = ttk.LabelFrame(self.matrix_frame, text="Input Matrix A", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        middle_frame = ttk.LabelFrame(self.matrix_frame, text="Weight Matrix B", padding="10")
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        right_frame = ttk.LabelFrame(self.matrix_frame, text="Result Matrix C", padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Configure grid weights
        self.matrix_frame.columnconfigure(0, weight=1)
        self.matrix_frame.columnconfigure(1, weight=1)
        self.matrix_frame.columnconfigure(2, weight=1)
        self.matrix_frame.rowconfigure(0, weight=1)
        
        # Create matrix entry widgets
        self.matrix_a_entries = []
        self.matrix_b_entries = []
        self.result_labels = []
        
        # Setup matrix entry grids
        self.setup_matrix_grid(left_frame, self.matrix_a_entries, "A")
        self.setup_matrix_grid(middle_frame, self.matrix_b_entries, "B")
        self.setup_result_grid(right_frame)
        
    def setup_matrix_grid(self, parent, entry_list, matrix_name):
        """Setup a grid of entry widgets for matrix input"""
        # Clear existing entries
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Entry):
                widget.destroy()
        entry_list.clear()
        
        size = int(self.size_var.get())
        
        for i in range(size):
            row_entries = []
            for j in range(size):
                entry = tk.Entry(parent, width=8, justify='center')
                entry.grid(row=i, column=j, padx=2, pady=2)
                entry.bind('<Return>', self.on_matrix_change)
                row_entries.append(entry)
            entry_list.append(row_entries)
            
        # Add load/save buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=size, column=0, columnspan=size, pady=(10, 0))
        
        ttk.Button(button_frame, text=f"Load {matrix_name}", 
                  command=lambda: self.load_matrix(matrix_name)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text=f"Save {matrix_name}", 
                  command=lambda: self.save_matrix(matrix_name)).pack(side=tk.LEFT)
        
    def setup_result_grid(self, parent):
        """Setup a grid of labels for result display"""
        # Clear existing labels
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Label):
                widget.destroy()
        self.result_labels.clear()
        
        size = int(self.size_var.get())
        
        for i in range(size):
            row_labels = []
            for j in range(size):
                label = tk.Label(parent, width=10, height=2, relief='sunken', 
                               bg='white', text="0.000")
                label.grid(row=i, column=j, padx=2, pady=2)
                row_labels.append(label)
            self.result_labels.append(row_labels)
            
    def setup_computation_tab(self):
        """Setup the computation results tab"""
        self.comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comp_frame, text="Computation Results")
        
        # Create left panel for results and right panel for stats
        left_panel = ttk.LabelFrame(self.comp_frame, text="Computation Details", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        right_panel = ttk.LabelFrame(self.comp_frame, text="Performance Statistics", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.comp_frame.columnconfigure(0, weight=2)
        self.comp_frame.columnconfigure(1, weight=1)
        self.comp_frame.rowconfigure(0, weight=1)
        
        # Computation results text area
        self.comp_text = tk.Text(left_panel, height=20, width=50)
        comp_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=self.comp_text.yview)
        self.comp_text.configure(yscrollcommand=comp_scrollbar.set)
        
        self.comp_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        comp_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)
        
        # Performance statistics
        self.stats_tree = ttk.Treeview(right_panel, columns=('Value',), show='tree headings', height=15)
        self.stats_tree.heading('#0', text='Metric')
        self.stats_tree.heading('Value', text='Value')
        self.stats_tree.column('#0', width=200)
        self.stats_tree.column('Value', width=100)
        
        stats_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

    def setup_visualization_tab(self):
        """Set up the visualization tab"""

        # Create style
        style = ttk.Style()
        style.configure("Custom.TLabel", font=("Space Mono", 10))
        style.configure("Custom.TEntry", font=("Space Mono", 10))
        style.configure("Custom.TButton", font=("Space Mono", 10))

        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")

        # Control panel for visualization
        viz_control = ttk.LabelFrame(self.viz_frame, text="Visualization Controls", padding="10")
        viz_control.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(viz_control, text="Cycle Range:", style="Custom.TLabel").grid(row=0, column=0, padx=(0, 5))
        self.cycle_start_var = tk.StringVar(value="0")
        ttk.Entry(viz_control, width=5, textvariable=self.cycle_start_var, style="Custom.TEntry").grid(row=0, column=1,
                                                                                                       padx=(0, 5))
        ttk.Label(viz_control, text="to", style="Custom.TLabel").grid(row=0, column=2, padx=5)
        self.cycle_end_var = tk.StringVar(value="5")
        ttk.Entry(viz_control, width=5, textvariable=self.cycle_end_var, style="Custom.TEntry").grid(row=0, column=3,
                                                                                                     padx=(0, 20))

        ttk.Button(viz_control, text="Update Visualization",
                   command=self.update_visualization, style="Custom.TButton").grid(row=0, column=4, padx=(10, 0))

        # Matplotlib canvas for visualization
        self.viz_fig, self.viz_ax = plt.subplots(figsize=(10, 6))
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, self.viz_frame)
        self.viz_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2,
                                             sticky=(tk.W, tk.E, tk.N, tk.S))

        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.rowconfigure(1, weight=1)
        
    def setup_benchmark_tab(self):
        """Set up the benchmark tab"""
        self.bench_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bench_frame, text="Benchmark")
        
        # Benchmark controls
        bench_control = ttk.LabelFrame(self.bench_frame, text="Benchmark Settings", padding="10")
        bench_control.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(bench_control, text="Size Range:").grid(row=0, column=0, padx=(0, 5))
        self.bench_start_var = tk.StringVar(value="2")
        ttk.Entry(bench_control, width=5, textvariable=self.bench_start_var).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(bench_control, text="to").grid(row=0, column=2, padx=5)
        self.bench_end_var = tk.StringVar(value="8")
        ttk.Entry(bench_control, width=5, textvariable=self.bench_end_var).grid(row=0, column=3, padx=(0, 10))
        
        self.bench_progress = ttk.Progressbar(bench_control, mode='indeterminate')
        self.bench_progress.grid(row=0, column=4, padx=(10, 5), sticky=(tk.W, tk.E))
        bench_control.columnconfigure(4, weight=1)
        
        # Benchmark results table
        bench_results_frame = ttk.LabelFrame(self.bench_frame, text="Benchmark Results", padding="10")
        bench_results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.bench_tree = ttk.Treeview(bench_results_frame, show='headings', height=15)
        bench_tree_scroll = ttk.Scrollbar(bench_results_frame, orient="vertical", 
                                        command=self.bench_tree.yview)
        self.bench_tree.configure(yscrollcommand=bench_tree_scroll.set)
        
        self.bench_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        bench_tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        bench_results_frame.columnconfigure(0, weight=1)
        bench_results_frame.rowconfigure(0, weight=1)
        
        # Benchmark visualization
        bench_viz_frame = ttk.LabelFrame(self.bench_frame, text="Performance Plots", padding="10")
        bench_viz_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.bench_fig, self.bench_axes = plt.subplots(2, 2, figsize=(8, 6))
        self.bench_canvas = FigureCanvasTkAgg(self.bench_fig, bench_viz_frame)
        self.bench_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        bench_viz_frame.columnconfigure(0, weight=1)
        bench_viz_frame.rowconfigure(0, weight=1)
        
        self.bench_frame.columnconfigure(0, weight=1)
        self.bench_frame.columnconfigure(1, weight=1)
        self.bench_frame.rowconfigure(1, weight=1)
        
    def on_size_change(self, event=None):
        """Handle matrix size change"""
        try:
            new_size = int(self.size_var.get())
            if new_size < 2 or new_size > 10:
                messagebox.showerror("Error", "Matrix size must be between 2 and 10")
                self.size_var.set(str(self.current_size))
                return
                
            self.current_size = new_size
            self.setup_matrix_grid(self.matrix_frame.winfo_children()[0], self.matrix_a_entries, "A")
            self.setup_matrix_grid(self.matrix_frame.winfo_children()[1], self.matrix_b_entries, "B")
            self.setup_result_grid(self.matrix_frame.winfo_children()[2])
            self.generate_random_matrices()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer")
            self.size_var.set(str(self.current_size))
            
    def on_matrix_change(self, event=None):
        """Handle manual matrix entry changes"""
        self.read_matrices_from_gui()
        
    def generate_random_matrices(self):
        """Generate random matrices and update GUI"""
        size = int(self.size_var.get())
        self.matrix_a = np.random.randn(size, size).astype(np.float32)
        self.matrix_b = np.random.randn(size, size).astype(np.float32)
        self.update_matrix_display()
        
    def generate_identity_matrices(self):
        """Generate identity matrices and update GUI"""
        size = int(self.size_var.get())
        self.matrix_a = np.eye(size, dtype=np.float32)
        self.matrix_b = np.eye(size, dtype=np.float32)
        self.update_matrix_display()
        
    def generate_ones_matrices(self):
        """Generate ones matrices and update GUI"""
        size = int(self.size_var.get())
        self.matrix_a = np.ones((size, size), dtype=np.float32)
        self.matrix_b = np.ones((size, size), dtype=np.float32)
        self.update_matrix_display()
        
    def update_matrix_display(self):
        """Update the matrix display in GUI"""
        if self.matrix_a is not None:
            for i in range(len(self.matrix_a_entries)):
                for j in range(len(self.matrix_a_entries[i])):
                    self.matrix_a_entries[i][j].delete(0, tk.END)
                    self.matrix_a_entries[i][j].insert(0, f"{self.matrix_a[i, j]:.3f}")
                    
        if self.matrix_b is not None:
            for i in range(len(self.matrix_b_entries)):
                for j in range(len(self.matrix_b_entries[i])):
                    self.matrix_b_entries[i][j].delete(0, tk.END)
                    self.matrix_b_entries[i][j].insert(0, f"{self.matrix_b[i, j]:.3f}")
                    
    def read_matrices_from_gui(self):
        """Read matrices from GUI entries"""
        try:
            size = int(self.size_var.get())
            
            # Read matrix A
            matrix_a = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    val = self.matrix_a_entries[i][j].get()
                    matrix_a[i, j] = float(val) if val else 0.0
            self.matrix_a = matrix_a
            
            # Read matrix B
            matrix_b = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    val = self.matrix_b_entries[i][j].get()
                    matrix_b[i, j] = float(val) if val else 0.0
            self.matrix_b = matrix_b
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid matrix values: {e}")
            
    def update_result_display(self):
        """Update the result matrix display"""
        if self.result_matrix is not None:
            size = min(len(self.result_labels), self.result_matrix.shape[0])
            for i in range(size):
                for j in range(min(len(self.result_labels[i]), self.result_matrix.shape[1])):
                    self.result_labels[i][j].config(text=f"{self.result_matrix[i, j]:.3f}")
                    
    def run_computation(self):
        """Run the systolic array computation"""
        try:
            self.read_matrices_from_gui()
            
            if self.matrix_a is None or self.matrix_b is None:
                messagebox.showerror("Error", "Please set up matrices first")
                return
                
            # Clear previous results
            self.comp_text.delete(1.0, tk.END)
            
            # Create systolic array
            array_size = int(self.array_size_var.get())
            self.systolic_array = SystolicArray(array_size, array_size)
            
            # Run computation
            self.comp_text.insert(tk.END, "Running systolic array computation...\n")
            self.comp_text.insert(tk.END, f"Matrix A shape: {self.matrix_a.shape}\n")
            self.comp_text.insert(tk.END, f"Matrix B shape: {self.matrix_b.shape}\n")
            self.comp_text.insert(tk.END, f"Array size: {array_size}x{array_size}\n\n")
            
            start_time = time.time()
            result, cycles = self.systolic_array.matrix_multiply(self.matrix_a, self.matrix_b, verbose=False)
            end_time = time.time()
            
            # Extract relevant portion
            size = self.matrix_a.shape[0]
            self.result_matrix = result[:size, :size]
            
            # Update result display
            self.update_result_display()
            
            # Calculate expected result and error
            expected = np.dot(self.matrix_a, self.matrix_b)
            error = np.max(np.abs(expected - self.result_matrix))
            
            # Update computation text
            self.comp_text.insert(tk.END, f"Computation completed in {cycles} cycles\n")
            self.comp_text.insert(tk.END, f"Wall clock time: {end_time - start_time:.4f} seconds\n")
            self.comp_text.insert(tk.END, f"Maximum error: {error:.10f}\n\n")
            
            self.comp_text.insert(tk.END, "Systolic Array Result:\n")
            self.comp_text.insert(tk.END, str(self.result_matrix) + "\n\n")
            
            self.comp_text.insert(tk.END, "Expected Result (NumPy):\n")
            self.comp_text.insert(tk.END, str(expected) + "\n\n")
            
            # Update statistics
            self.update_statistics(cycles, error, end_time - start_time)
            
            # Update visualization
            self.update_visualization()
            
            messagebox.showinfo("Success", f"Computation completed in {cycles} cycles")
            
        except Exception as e:
            messagebox.showerror("Error", f"Computation failed: {e}")
            
    def update_statistics(self, cycles, error, wall_time):
        """Update the statistics tree"""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
            
        # Add new statistics
        self.stats_tree.insert('', 'end', text='Cycles', values=(cycles,))
        self.stats_tree.insert('', 'end', text='Wall Time (s)', values=(f"{wall_time:.4f}",))
        self.stats_tree.insert('', 'end', text='Max Error', values=(f"{error:.2e}",))
        
        if self.matrix_a is not None:
            theoretical_ops = self.matrix_a.shape[0] ** 3 * 2
            self.stats_tree.insert('', 'end', text='Theoretical Ops', values=(theoretical_ops,))
            
            if cycles > 0:
                ops_per_cycle = theoretical_ops / cycles
                self.stats_tree.insert('', 'end', text='Ops/Cycle', values=(f"{ops_per_cycle:.2f}",))
                
            if wall_time > 0:
                ops_per_sec = theoretical_ops / wall_time
                self.stats_tree.insert('', 'end', text='Ops/Second', values=(f"{ops_per_sec:.0f}",))
                
    def update_visualization(self):
        """Update the systolic array visualization"""
        if self.systolic_array is None or not self.systolic_array.computation_history:
            return
            
        try:
            start_cycle = int(self.cycle_start_var.get())
            end_cycle = int(self.cycle_end_var.get())
            
            # Clear previous plot
            self.viz_fig.clear()
            
            # Create visualization using the existing function
            history_len = len(self.systolic_array.computation_history)
            end_cycle = min(end_cycle, history_len)
            
            if end_cycle <= start_cycle:
                self.viz_ax = self.viz_fig.add_subplot(111)
                self.viz_ax.text(0.5, 0.5, 'No cycles to display', 
                               ha='center', va='center', transform=self.viz_ax.transAxes)
                self.viz_canvas.draw()
                return
                
            num_plots = min(5, end_cycle - start_cycle)
            
            for idx, cycle in enumerate(range(start_cycle, start_cycle + num_plots)):
                if cycle >= history_len:
                    break
                    
                ax = self.viz_fig.add_subplot(1, num_plots, idx + 1)
                self.plot_cycle_state(ax, cycle)
                
            self.viz_fig.tight_layout()
            self.viz_canvas.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid cycle numbers")
            
    def plot_cycle_state(self, ax, cycle):
        """Plot the state of the systolic array for a specific cycle"""
        if cycle >= len(self.systolic_array.computation_history):
            return
            
        cycle_data = self.systolic_array.computation_history[cycle]
        
        # Create heatmap data
        pe_values = np.zeros((self.systolic_array.rows, self.systolic_array.cols))
        
        # Fill in data from active PEs
        for pe_state in cycle_data['pe_states']:
            row, col = pe_state['row'], pe_state['col']
            if row < self.systolic_array.rows and col < self.systolic_array.cols:
                pe_values[row, col] = pe_state['accumulator'] if pe_state['accumulator'] else 0
                
        # Plot heatmap
        im = ax.imshow(pe_values, cmap='viridis', aspect='equal')
        ax.set_title(f'Cycle {cycle}')
        ax.set_xlabel('PE Column')
        ax.set_ylabel('PE Row')
        
        # Add text annotations
        for pe_state in cycle_data['pe_states']:
            row, col = pe_state['row'], pe_state['col']
            if row < self.systolic_array.rows and col < self.systolic_array.cols:
                input_val = pe_state['input'] if pe_state['input'] is not None else 0
                weight = pe_state['weight']
                acc = pe_state['accumulator'] if pe_state['accumulator'] else 0
                
                text = f"I:{input_val:.1f}\nW:{weight:.1f}\nA:{acc:.1f}"
                ax.text(
                    col, row, text, ha='center', va='center', fontsize=12, color='black', font="Space Mono", fontweight="bold",
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

    def run_benchmark(self):
        """Run benchmark in a separate thread"""
        def benchmark_thread():
            try:
                start_size = int(self.bench_start_var.get())
                end_size = int(self.bench_end_var.get())
                
                if start_size >= end_size or start_size < 2 or end_size > 12:
                    self.result_queue.put(("error", "Invalid size range. Use 2-12."))
                    return
                    
                sizes = list(range(start_size, end_size + 1))
                
                self.result_queue.put(("progress", "start"))
                
                comparator = CPUSystolicComparison()
                array_size = int(self.array_size_var.get())
                
                results = comparator.benchmark_comparison(sizes, (array_size, array_size))
                
                self.result_queue.put(("progress", "stop"))
                self.result_queue.put(("benchmark_results", results))
                
            except Exception as e:
                self.result_queue.put(("progress", "stop"))
                self.result_queue.put(("error", str(e)))
                
        # Start benchmark thread
        threading.Thread(target=benchmark_thread, daemon=True).start()
        
        # Start checking for results
        self.check_benchmark_results()
        
    def check_benchmark_results(self):
        """Check for benchmark results from the thread"""
        try:
            while True:
                result_type, data = self.result_queue.get_nowait()
                
                if result_type == "progress":
                    if data == "start":
                        self.bench_progress.start()
                    elif data == "stop":
                        self.bench_progress.stop()
                        
                elif result_type == "benchmark_results":
                    self.benchmark_results = data
                    self.update_benchmark_display()
                    self.plot_benchmark_results()
                    messagebox.showinfo("Success", "Benchmark completed successfully!")
                    
                elif result_type == "error":
                    messagebox.showerror("Benchmark Error", data)
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.check_benchmark_results)
        
    def update_benchmark_display(self):
        """Update the benchmark results table"""
        if self.benchmark_results is None:
            return
            
        # Clear existing items
        for item in self.bench_tree.get_children():
            self.bench_tree.delete(item)
            
        # Set up columns
        columns = ['Size', 'CPU Naive (s)', 'CPU NumPy (s)', 'Systolic Cycles', 'Error', 'Throughput']
        self.bench_tree['columns'] = columns
        self.bench_tree['show'] = 'headings'
        
        for col in columns:
            self.bench_tree.heading(col, text=col)
            self.bench_tree.column(col, width=100)
            
        # Add data
        for _, row in self.benchmark_results.iterrows():
            values = [
                f"{row['matrix_size']}",
                f"{row['cpu_naive_time']:.4f}" if pd.notna(row['cpu_naive_time']) else "N/A",
                f"{row['cpu_numpy_time']:.4f}" if pd.notna(row['cpu_numpy_time']) else "N/A",
                f"{row['systolic_cycles']}" if pd.notna(row['systolic_cycles']) else "N/A",
                f"{row['systolic_error']:.2e}" if pd.notna(row['systolic_error']) else "N/A",
                f"{row['systolic_throughput']:.2f}" if pd.notna(row['systolic_throughput']) else "N/A"
            ]
            self.bench_tree.insert('', 'end', values=values)
            
    def plot_benchmark_results(self):
        """Plot benchmark results"""
        if self.benchmark_results is None:
            return
            
        # Clear previous plots
        for ax in self.bench_axes.flat:
            ax.clear()
            
        # Filter valid data
        valid_data = self.benchmark_results.dropna(subset=['systolic_cycles'])
        
        # Plot 1: Execution time comparison
        ax1 = self.bench_axes[0, 0]
        ax1.plot(self.benchmark_results['matrix_size'], self.benchmark_results['cpu_naive_time'], 
                'o-', label='CPU Naive', linewidth=2)
        ax1.plot(self.benchmark_results['matrix_size'], self.benchmark_results['cpu_numpy_time'], 
                's-', label='CPU NumPy', linewidth=2)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cycles vs Operations
        ax2 = self.bench_axes[0, 1]
        if not valid_data.empty:
            ax2.plot(valid_data['matrix_size'], valid_data['systolic_cycles'], 
                    'o-', label='Systolic Cycles', color='red', linewidth=2)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(valid_data['matrix_size'], valid_data['cpu_numpy_ops'], 
                         's-', label='CPU Operations', color='blue', linewidth=2)
            ax2.set_xlabel('Matrix Size')
            ax2.set_ylabel('Systolic Cycles', color='red')
            ax2_twin.set_ylabel('CPU Operations', color='blue')
            ax2.set_title('Cycles vs Operations')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput comparison
        ax3 = self.bench_axes[1, 0]
        if not valid_data.empty:
            cpu_throughput = valid_data['cpu_numpy_ops'] / valid_data['cpu_numpy_time']
            ax3.plot(valid_data['matrix_size'], cpu_throughput, 
                    'o-', label='CPU Throughput (ops/sec)', linewidth=2)
            ax3.plot(valid_data['matrix_size'], valid_data['systolic_throughput'], 
                    's-', label='Systolic Throughput (ops/cycle)', linewidth=2)
            ax3.set_xlabel('Matrix Size')
            ax3.set_ylabel('Throughput')
            ax3.set_title('Throughput Comparison')
            ax3.legend()
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        ax4 = self.bench_axes[1, 1]
        if not valid_data.empty:
            ax4.plot(valid_data['matrix_size'], valid_data['systolic_error'], 
                    'o-', color='green', linewidth=2)
            ax4.set_xlabel('Matrix Size')
            ax4.set_ylabel('Maximum Error')
            ax4.set_title('Numerical Error')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        self.bench_fig.tight_layout()
        self.bench_canvas.draw()
        
    def load_matrix(self, matrix_name):
        """Load matrix from file"""
        filename = filedialog.askopenfilename(
            title=f"Load Matrix {matrix_name}",
            filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    matrix = np.load(filename)
                else:
                    matrix = np.loadtxt(filename, delimiter=',')
                
                # Check size compatibility
                if matrix.shape[0] != int(self.size_var.get()) or matrix.shape[1] != int(self.size_var.get()):
                    response = messagebox.askyesno(
                        "Size Mismatch", 
                        f"Matrix size {matrix.shape} doesn't match current size {self.size_var.get()}. "
                        f"Resize to match matrix?"
                    )
                    if response:
                        self.size_var.set(str(matrix.shape[0]))
                        self.on_size_change()
                    else:
                        return
                
                if matrix_name == "A":
                    self.matrix_a = matrix.astype(np.float32)
                else:
                    self.matrix_b = matrix.astype(np.float32)
                    
                self.update_matrix_display()
                messagebox.showinfo("Success", f"Matrix {matrix_name} loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load matrix: {e}")
                
    def save_matrix(self, matrix_name):
        """Save matrix to file"""
        self.read_matrices_from_gui()
        
        matrix = self.matrix_a if matrix_name == "A" else self.matrix_b
        if matrix is None:
            messagebox.showerror("Error", f"Matrix {matrix_name} is empty")
            return
            
        filename = filedialog.asksaveasfilename(
            title=f"Save Matrix {matrix_name}",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    np.save(filename, matrix)
                else:
                    np.savetxt(filename, matrix, delimiter=',', fmt='%.6f')
                    
                messagebox.showinfo("Success", f"Matrix {matrix_name} saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save matrix: {e}")


def main():
    """Main function to start the GUI application"""
    root = tk.Tk()
    tkFont.nametofont("TkDefaultFont").configure(family="Space Mono")
    tkFont.nametofont("TkTextFont").configure(family="Space Mono")
    tkFont.nametofont("TkFixedFont").configure(family="Space Mono")

    # Configure ttk styles
    style = ttk.Style()
    
    # Try to use a modern theme
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'alt' in available_themes:
        style.theme_use('alt')
    
    # Configure custom styles
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    # Create and run the application
    app = SystolicArrayGUI(root)
    
    # Handle window closing
    def on_closing():
        plt.close('all')  # Close all matplotlib windows
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()