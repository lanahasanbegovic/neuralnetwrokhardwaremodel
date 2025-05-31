import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import queue
import time
import logging
from NNHM import (
    SystolicArray,
    CPUSystolicComparison,
    visualize_systolic_array_operation,
    plot_performance_comparison
)
import tkinter.font as tkFont

def configure_fonts():
    """Configure fonts for the entire application"""
    font_family = 'Space Mono'
    plt.rcParams['font.family'] = font_family

    # Configure tkinter fonts
    for font_name in ["TkDefaultFont", "TkTextFont", "TkFixedFont"]:
        tkFont.nametofont(font_name).configure(family=font_family)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('systolic_array_gui.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )

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

        # Cache related
        self._result_cache = {}  # Cache computed results
        self._max_cache_size = 10

        # Initialize GUI variables early
        self.size_var = tk.StringVar(value="3")
        self.array_size_var = tk.StringVar(value="8")

        # ADD THESE NEW THREADING ATTRIBUTES:
        self.benchmark_event = threading.Event()
        self.benchmark_thread = None
        self.benchmark_completed = False
        self._last_benchmark_hash = None

        # Create queue for thread communication (keep existing)
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

    def resize_matrix_grid(self, parent, entry_list, matrix_name):
        """Resize existing matrix grid instead of recreating"""
        current_size = len(entry_list) if entry_list else 0
        target_size = int(self.size_var.get())

        if current_size == target_size:
            return  # No change needed

        # If we need to shrink the grid
        if target_size < current_size:
            # Remove excess rows
            for i in range(target_size, current_size):
                for entry in entry_list[i]:
                    entry.destroy()
            # Trim the entry_list
            entry_list[:] = entry_list[:target_size]

            # Remove excess columns from remaining rows
            for i in range(target_size):
                for j in range(target_size, len(entry_list[i])):
                    entry_list[i][j].destroy()
                # Trim each row
                entry_list[i][:] = entry_list[i][:target_size]

        # If we need to expand the grid
        elif target_size > current_size:
            # First, expand existing rows (add columns)
            for i in range(current_size):
                for j in range(len(entry_list[i]), target_size):
                    entry = tk.Entry(parent, width=8, justify='center')
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    entry.bind('<Return>', self.on_matrix_change)
                    entry_list[i].append(entry)

            # Then add new rows
            for i in range(current_size, target_size):
                row_entries = []
                for j in range(target_size):
                    entry = tk.Entry(parent, width=8, justify='center')
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    entry.bind('<Return>', self.on_matrix_change)
                    row_entries.append(entry)
                entry_list.append(row_entries)

        # Update or create the button frame
        self.update_matrix_buttons(parent, matrix_name, target_size)

    def update_matrix_buttons(self, parent, matrix_name, grid_size):
        """Update the load/save buttons below the matrix grid"""
        # Find and remove existing button frame
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.Frame):
                # Check if this frame contains buttons by looking for button children
                has_buttons = any(isinstance(child, ttk.Button) for child in widget.winfo_children())
                if has_buttons:
                    widget.destroy()
                    break

        # Create new button frame
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=grid_size, column=0, columnspan=grid_size, pady=(10, 0))

        ttk.Button(button_frame, text=f"Load {matrix_name}",
                   command=lambda: self.load_matrix(matrix_name)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text=f"Save {matrix_name}",
                   command=lambda: self.save_matrix(matrix_name)).pack(side=tk.LEFT)

    def setup_matrix_grid(self, parent, entry_list, matrix_name):
        """Setup a grid of entry widgets for matrix input - now uses resize logic"""
        # Use the new resize method instead of recreating everything
        self.resize_matrix_grid(parent, entry_list, matrix_name)

    def on_size_change(self, event=None):
        """Handle matrix size change - UPDATED to use resize method"""
        try:
            new_size = int(self.size_var.get())
            if new_size < 2 or new_size > 10:
                messagebox.showerror("Error", "Matrix size must be between 2 and 10")
                self.size_var.set(str(self.current_size))
                return

            self.current_size = new_size

            # Use resize instead of complete recreation
            self.resize_matrix_grid(
                self.matrix_frame.winfo_children()[0],
                self.matrix_a_entries,
                "A"
            )
            self.resize_matrix_grid(
                self.matrix_frame.winfo_children()[1],
                self.matrix_b_entries,
                "B"
            )
            self.setup_result_grid(self.matrix_frame.winfo_children()[2])

            # Generate new matrices for the new size
            self.generate_random_matrices()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer")
            self.size_var.set(str(self.current_size))

    # ALSO ADD this optimized matrix display update method:
    def update_matrix_display(self):
        """Update the matrix display in GUI - optimized version"""
        if self.matrix_a is not None and self.matrix_a_entries:
            for i in range(min(len(self.matrix_a_entries), self.matrix_a.shape[0])):
                for j in range(min(len(self.matrix_a_entries[i]), self.matrix_a.shape[1])):
                    new_value = f"{self.matrix_a[i, j]:.3f}"
                    current_value = self.matrix_a_entries[i][j].get()
                    if current_value != new_value:  # Only update if changed
                        self.matrix_a_entries[i][j].delete(0, tk.END)
                        self.matrix_a_entries[i][j].insert(0, new_value)

        if self.matrix_b is not None and self.matrix_b_entries:
            for i in range(min(len(self.matrix_b_entries), self.matrix_b.shape[0])):
                for j in range(min(len(self.matrix_b_entries[i]), self.matrix_b.shape[1])):
                    new_value = f"{self.matrix_b[i, j]:.3f}"
                    current_value = self.matrix_b_entries[i][j].get()
                    if current_value != new_value:  # Only update if changed
                        self.matrix_b_entries[i][j].delete(0, tk.END)
                        self.matrix_b_entries[i][j].insert(0, new_value)

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

        # After comp_text and scrollbar
        clear_button = ttk.Button(left_panel, text="Clear Results", command=self.clear_computation_results)
        clear_button.grid(row=1, column=0, columnspan=2, pady=(5, 0))

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

    def clear_computation_results(self):
        """Clear computation results and statistics display"""
        self.comp_text.delete(1.0, tk.END)
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        self.result_matrix = None
        self.update_result_display()

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

        columns = ['Size', 'CPU Naive (s)', 'CPU NumPy (s)', 'Systolic Cycles', 'Error', 'Throughput']
        self.bench_tree = ttk.Treeview(bench_results_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.bench_tree.heading(col, text=col)
            self.bench_tree.column(col, width=100)

        bench_tree_scroll = ttk.Scrollbar(bench_results_frame, orient="vertical",
                                          command=self.bench_tree.yview)
        self.bench_tree.configure(yscrollcommand=bench_tree_scroll.set)

        self.bench_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        bench_tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        clear_bench_button = ttk.Button(bench_results_frame, text="Clear Results", command=self.clear_benchmark_results)
        clear_bench_button.grid(row=1, column=0, columnspan=2, pady=(5, 0))

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

    def clear_benchmark_results(self):
        """Clear benchmark results and plots"""
        for item in self.bench_tree.get_children():
            self.bench_tree.delete(item)
        self.benchmark_results = None
        for ax in self.bench_axes.flat:
            ax.clear()
        self.bench_fig.tight_layout()
        self.bench_canvas.draw_idle()

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
        """Run the systolic array computation with caching and improved error handling"""
        try:
            # Read matrices from GUI
            self.read_matrices_from_gui()

            if self.matrix_a is None or self.matrix_b is None:
                messagebox.showerror("Input Error", "Please set up matrices first")
                return

            # Validate matrix dimensions
            if self.matrix_a.shape != self.matrix_b.shape:
                messagebox.showerror("Input Error",
                                     f"Matrix dimensions don't match: A{self.matrix_a.shape} vs B{self.matrix_b.shape}")
                return

            # Validate array size
            array_size = int(self.array_size_var.get())
            if array_size < 2 or array_size > 32:
                messagebox.showerror("Input Error", "Array size must be between 2 and 32")
                return

            # Generate cache key
            cache_key = self._generate_cache_key(self.matrix_a, self.matrix_b, array_size)

            # Check if result is already cached
            if cache_key in self._result_cache:
                print("Using cached result!")  # Debug message
                cached_result = self._result_cache[cache_key]
                self.result_matrix = cached_result['result']
                cycles = cached_result['cycles']

                self.update_result_display()

                expected = np.dot(self.matrix_a, self.matrix_b)
                error = np.max(np.abs(expected - self.result_matrix))

                self.comp_text.delete(1.0, tk.END)
                self.comp_text.insert(tk.END, "Using cached computation result...\n")
                self.comp_text.insert(tk.END, f"Matrix A shape: {self.matrix_a.shape}\n")
                self.comp_text.insert(tk.END, f"Matrix B shape: {self.matrix_b.shape}\n")
                self.comp_text.insert(tk.END, f"Array size: {array_size}x{array_size}\n\n")
                self.comp_text.insert(tk.END, f"Cached result: {cycles} cycles\n")
                self.comp_text.insert(tk.END, f"Maximum error: {error:.10f}\n\n")

                self.update_statistics(cycles, error, 0.0)
                self.update_visualization()

                messagebox.showinfo("Success", f"Used cached result: {cycles} cycles")
                return

            # Warn about large matrix
            max_size = 20
            if self.matrix_a.shape[0] > max_size:
                response = messagebox.askyesno("Large Matrix Warning",
                                               f"Matrix size {self.matrix_a.shape[0]} is large. This may take time and memory. Continue?")
                if not response:
                    return

            # Clear previous results
            self.comp_text.delete(1.0, tk.END)

            # Create systolic array
            self.systolic_array = SystolicArray(array_size, array_size)

            # Log computation start
            logging.info(
                f"Starting computation: Matrix size {self.matrix_a.shape}, Array size {array_size}x{array_size}")

            self.comp_text.insert(tk.END, "Running systolic array computation...\n")
            self.comp_text.insert(tk.END, f"Matrix A shape: {self.matrix_a.shape}\n")
            self.comp_text.insert(tk.END, f"Matrix B shape: {self.matrix_b.shape}\n")
            self.comp_text.insert(tk.END, f"Array size: {array_size}x{array_size}\n\n")
            self.comp_text.update()

            start_time = time.time()
            result, cycles = self.systolic_array.matrix_multiply(self.matrix_a, self.matrix_b, verbose=False)
            end_time = time.time()

            if result is None:
                raise RuntimeError("Systolic array computation returned None result")
            if cycles <= 0:
                raise RuntimeError(f"Invalid cycle count: {cycles}")

            size = self.matrix_a.shape[0]
            self.result_matrix = result[:size, :size]

            # Cache the result
            self._result_cache[cache_key] = {
                'result': self.result_matrix.copy(),
                'cycles': cycles,
                'timestamp': time.time()
            }
            self.clear_old_cache()

            self.update_result_display()

            expected = np.dot(self.matrix_a, self.matrix_b)
            error = np.max(np.abs(expected - self.result_matrix))

            if np.isnan(error) or np.isinf(error):
                raise RuntimeError("Computation produced NaN or infinite values")

            self.comp_text.insert(tk.END, f"Computation completed in {cycles} cycles\n")
            self.comp_text.insert(tk.END, f"Wall clock time: {end_time - start_time:.4f} seconds\n")
            self.comp_text.insert(tk.END, f"Maximum error: {error:.10f}\n")
            self.comp_text.insert(tk.END, f"Result cached for future use\n\n")
            self.comp_text.insert(tk.END, "Systolic Array Result:\n")
            self.comp_text.insert(tk.END, str(self.result_matrix) + "\n\n")
            self.comp_text.insert(tk.END, "Expected Result (NumPy):\n")
            self.comp_text.insert(tk.END, str(expected) + "\n\n")

            self.update_statistics(cycles, error, end_time - start_time)
            self.update_visualization()

            logging.info(f"Computation completed successfully: {cycles} cycles, error: {error:.2e}")
            messagebox.showinfo("Success", f"Computation completed in {cycles} cycles\nMax error: {error:.2e}")

        except ValueError as e:
            error_msg = f"Invalid input values: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Input Error", error_msg)

        except MemoryError as e:
            error_msg = "Insufficient memory for computation. Try smaller matrices or array size."
            logging.error(f"Memory error: {e}")
            messagebox.showerror("Memory Error", error_msg)

        except RuntimeError as e:
            error_msg = f"Computation runtime error: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Runtime Error", error_msg)

        except ImportError as e:
            error_msg = f"Missing required module: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Import Error", error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during computation: {str(e)}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Unexpected Error", error_msg)

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

            history_len = len(self.systolic_array.computation_history)
            end_cycle = min(end_cycle, history_len)

            if end_cycle <= start_cycle:
                self.viz_fig.clear()
                self.viz_ax = self.viz_fig.add_subplot(111)
                self.viz_ax.text(0.5, 0.5, 'No cycles to display',
                                 ha='center', va='center', transform=self.viz_ax.transAxes)
                self.viz_canvas.draw_idle()
                return

            num_plots = end_cycle - start_cycle
            self.viz_fig.clear()  # Clear once

            # Determine layout
            cols = min(5, num_plots)  # max 5 columns for readability
            rows = (num_plots + cols - 1) // cols  # ceiling division

            for idx, cycle in enumerate(range(start_cycle, end_cycle)):
                if cycle >= history_len:
                    break
                ax = self.viz_fig.add_subplot(rows, cols, idx + 1)
                self.plot_cycle_state(ax, cycle)

            self.viz_fig.tight_layout()
            self.viz_canvas.draw_idle()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid cycle numbers")

    def plot_cycle_state(self, ax, cycle):
        """Plot the state of the systolic array for a specific cycle - OPTIMIZED"""
        if cycle >= len(self.systolic_array.computation_history):
            return

        cycle_data = self.systolic_array.computation_history[cycle]

        # Pre-allocate array (more efficient than zeros in loop)
        pe_values = np.zeros((self.systolic_array.rows, self.systolic_array.cols))

        # Store text annotations to add later (avoid repeated ax.text calls)
        annotations = []

        # Vectorized filling instead of nested conditions
        for pe_state in cycle_data['pe_states']:
            row, col = pe_state['row'], pe_state['col']

            # Bounds checking optimization
            if 0 <= row < self.systolic_array.rows and 0 <= col < self.systolic_array.cols:
                # Use .get() with default to avoid KeyError
                pe_values[row, col] = pe_state.get('accumulator', 0)

                # Prepare annotation data
                input_val = pe_state.get('input', 0) or 0
                weight = pe_state.get('weight', 0)
                acc = pe_state.get('accumulator', 0) or 0

                # Store annotation info for batch processing
                annotations.append({
                    'row': row, 'col': col,
                    'input': input_val, 'weight': weight, 'acc': acc
                })

        # Create heatmap with optimization flags
        im = ax.imshow(pe_values, cmap='viridis', aspect='equal',
                       animated=True, interpolation='nearest')

        # Set labels and title efficiently
        ax.set_title(f'Cycle {cycle}', fontsize=10)
        ax.set_xlabel('PE Column', fontsize=8)
        ax.set_ylabel('PE Row', fontsize=8)

        # Batch add annotations (more efficient than individual ax.text calls)
        for ann in annotations:
            text = f"I:{ann['input']:.1f}\nW:{ann['weight']:.1f}\nA:{ann['acc']:.1f}"
            ax.text(
                ann['col'], ann['row'], text,
                ha='center', va='center', fontsize=8,
                color='black', family='Space Mono', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

        # Optimize tick labels for better performance
        ax.set_xticks(range(self.systolic_array.cols))
        ax.set_yticks(range(self.systolic_array.rows))

        return im  # Return the image for potential animation use

    def run_benchmark(self):
        """Run benchmark with better thread management"""
        # Prevent multiple concurrent benchmarks
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            messagebox.showwarning("Warning", "Benchmark already running!")
            return

        # Reset completion flag
        self.benchmark_completed = False
        self.benchmark_event.clear()

        # Start the benchmark worker thread
        self.benchmark_thread = threading.Thread(target=self.benchmark_worker)
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()

        # Start progress indicator
        self.bench_progress.start()

        # Use event-driven updates instead of continuous polling
        self.root.after(100, self.check_benchmark_completion)

    def benchmark_worker(self):
        """Worker method that runs the actual benchmark in a separate thread"""
        try:
            start_size = int(self.bench_start_var.get())
            end_size = int(self.bench_end_var.get())

            if start_size >= end_size or start_size < 2 or end_size > 12:
                self.result_queue.put(("error", "Invalid size range. Use 2-12."))
                return

            sizes = list(range(start_size, end_size + 1))

            # Run the actual benchmark
            comparator = CPUSystolicComparison()
            array_size = int(self.array_size_var.get())
            results = comparator.benchmark_comparison(sizes, (array_size, array_size))

            # Put results in queue for main thread to process
            self.result_queue.put(("benchmark_results", results))

        except Exception as e:
            self.result_queue.put(("error", str(e)))
        finally:
            # Signal that benchmark is complete
            self.benchmark_completed = True
            self.benchmark_event.set()

    def check_benchmark_completion(self):
        """Check if benchmark is complete and process results"""
        # Process any available results from the queue
        try:
            while True:
                result_type, data = self.result_queue.get_nowait()

                if result_type == "benchmark_results":
                    self.benchmark_results = data
                    self.update_benchmark_display()
                    self.plot_benchmark_results()
                    self.bench_progress.stop()
                    messagebox.showinfo("Success", "Benchmark completed successfully!")
                    return  # Stop checking since we're done

                elif result_type == "error":
                    self.bench_progress.stop()
                    messagebox.showerror("Benchmark Error", data)
                    return  # Stop checking since we encountered an error

        except queue.Empty:
            pass  # No results yet, continue checking

        # If benchmark is still running, check again soon
        if not self.benchmark_completed:
            self.root.after(100, self.check_benchmark_completion)
        else:
            # Benchmark finished, stop progress indicator
            self.bench_progress.stop()

    def update_benchmark_display(self):
        """Update the benchmark results table"""
        if self.benchmark_results is None:
            return

        # Clear existing items
        for item in self.bench_tree.get_children():
            self.bench_tree.delete(item)

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
        """Optimized benchmark plotting with caching and efficient updates"""
        if self.benchmark_results is None:
            return

        # Check if data has changed to avoid unnecessary replotting
        try:
            current_hash = hash(str(self.benchmark_results.values.tobytes()))
            if hasattr(self, '_last_benchmark_hash') and current_hash == self._last_benchmark_hash:
                return  # Data hasn't changed, skip replotting
            self._last_benchmark_hash = current_hash
        except Exception:
            # Fallback if hashing fails
            pass

        # Clear previous plots efficiently
        for ax in self.bench_axes.flat:
            ax.clear()

        # Filter valid data once
        valid_data = self.benchmark_results.dropna(subset=['systolic_cycles'])

        if valid_data.empty:
            # Show "No data" message if no valid results
            self.bench_axes[0, 0].text(0.5, 0.5, 'No benchmark data available',
                                       ha='center', va='center', transform=self.bench_axes[0, 0].transAxes)
            self.bench_fig.tight_layout()
            self.bench_canvas.draw_idle()
            return

        # Pre-calculate common values to avoid repeated computation
        matrix_sizes = self.benchmark_results['matrix_size'].values
        valid_sizes = valid_data['matrix_size'].values

        # Plot 1: Execution time comparison
        ax1 = self.bench_axes[0, 0]
        cpu_naive_times = self.benchmark_results['cpu_naive_time'].dropna()
        cpu_numpy_times = self.benchmark_results['cpu_numpy_time'].dropna()

        if not cpu_naive_times.empty:
            naive_sizes = self.benchmark_results.loc[cpu_naive_times.index, 'matrix_size']
            ax1.plot(naive_sizes, cpu_naive_times, 'o-', label='CPU Naive',
                     linewidth=2, markersize=6)

        if not cpu_numpy_times.empty:
            numpy_sizes = self.benchmark_results.loc[cpu_numpy_times.index, 'matrix_size']
            ax1.plot(numpy_sizes, cpu_numpy_times, 's-', label='CPU NumPy',
                     linewidth=2, markersize=6)

        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cycles vs Operations
        ax2 = self.bench_axes[0, 1]
        if len(valid_data) > 0:
            # Primary y-axis: Systolic cycles
            line1 = ax2.plot(valid_sizes, valid_data['systolic_cycles'],
                             'o-', label='Systolic Cycles', color='red',
                             linewidth=2, markersize=6)
            ax2.set_xlabel('Matrix Size')
            ax2.set_ylabel('Systolic Cycles', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Secondary y-axis: CPU operations
            ax2_twin = ax2.twinx()
            if 'cpu_numpy_ops' in valid_data.columns:
                line2 = ax2_twin.plot(valid_sizes, valid_data['cpu_numpy_ops'],
                                      's-', label='CPU Operations', color='blue',
                                      linewidth=2, markersize=6)
                ax2_twin.set_ylabel('CPU Operations', color='blue')
                ax2_twin.tick_params(axis='y', labelcolor='blue')

            ax2.set_title('Cycles vs Operations')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Throughput comparison
        ax3 = self.bench_axes[1, 0]
        if len(valid_data) > 0:
            # Calculate CPU throughput efficiently
            cpu_throughput_data = self.benchmark_results[
                (self.benchmark_results['cpu_numpy_time'].notna()) &
                (self.benchmark_results['cpu_numpy_ops'].notna())
                ]

            if not cpu_throughput_data.empty:
                cpu_throughput = (cpu_throughput_data['cpu_numpy_ops'] /
                                  cpu_throughput_data['cpu_numpy_time'])
                ax3.plot(cpu_throughput_data['matrix_size'], cpu_throughput,
                         'o-', label='CPU Throughput (ops/sec)',
                         linewidth=2, markersize=6)

            # Systolic throughput
            systolic_throughput_data = valid_data['systolic_throughput'].dropna()
            if not systolic_throughput_data.empty:
                throughput_sizes = valid_data.loc[systolic_throughput_data.index, 'matrix_size']
                ax3.plot(throughput_sizes, systolic_throughput_data,
                         's-', label='Systolic Throughput (ops/cycle)',
                         linewidth=2, markersize=6)

            ax3.set_xlabel('Matrix Size')
            ax3.set_ylabel('Throughput')
            ax3.set_title('Throughput Comparison')
            ax3.legend()
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Error analysis
        ax4 = self.bench_axes[1, 1]
        error_data = valid_data['systolic_error'].dropna()
        if not error_data.empty:
            error_sizes = valid_data.loc[error_data.index, 'matrix_size']
            ax4.plot(error_sizes, error_data, 'o-', color='green',
                     linewidth=2, markersize=6)
            ax4.set_xlabel('Matrix Size')
            ax4.set_ylabel('Maximum Error')
            ax4.set_title('Numerical Error Analysis')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No error data available',
                     ha='center', va='center', transform=ax4.transAxes)

        # Apply tight layout and update canvas
        self.bench_fig.tight_layout()

        # Use draw_idle() instead of draw() for better performance
        self.bench_canvas.draw_idle()

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


    def clear_old_cache(self):
        """Clear old cached results to prevent memory leaks"""
        if len(self._result_cache) > self._max_cache_size:
            # Remove oldest entries (keep only the most recent max_cache_size entries)
            keys = list(self._result_cache.keys())
            for key in keys[:-self._max_cache_size]:
                del self._result_cache[key]

    def _generate_cache_key(self, matrix_a, matrix_b, array_size):
        """Generate a unique cache key for matrix computation"""
        # Create a hash based on matrix contents and array size
        a_hash = hash(matrix_a.tobytes())
        b_hash = hash(matrix_b.tobytes())
        return f"{a_hash}_{b_hash}_{array_size}"

def main():
    """Main function to start the GUI application"""
    setup_logging()
    logging.info("Starting Systolic Array GUI application")
    root = tk.Tk()
    configure_fonts()

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
        logging.info("Application interrupted by user")
        print("\nApplication interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        logging.info("Application shutting down")
        plt.close('all')


if __name__ == "__main__":
    main()