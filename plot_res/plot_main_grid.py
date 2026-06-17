"""
plot_main_grid.py

Generate the main paper figure: a 3x3 grid of BO performance curves across tasks.

Input
- Reads experiment result pickles from:
    <project_root>/exp_res_best/pkl/*.pkl
  Each pickle is expected to contain:
    best_all = [run_0, run_1, ...]
  where each run is a list of best-so-far objective values over iterations.

- Filenames are parsed as:
    <function>_<kernel_identifier>_<acq>.pkl

What is plotted
- For each task in PAPER_TASKS and each method in PARAMS_DICT, this script plots
  the mean trajectory across seeds (runs), optionally in:
    (i) log optimality gap (USE_LOG_GAP=True) if the global minimum is known, or
    (ii) raw best observed value (USE_LOG_GAP=False).

- Some tasks require sign-flip (NEGATE) or have different objective direction
  (e.g., portfolio is maximization).

Output
- Saves a PDF to:
    <project_root>/paper_figures/grid_errorbar_<acq>_<log_gap|raw_val>.pdf
Usage
  python plot_main_grid.py
  # optionally change ACQ_TO_PLOT inside the script (e.g., 'ei', 'ucb')
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict

#global switch: True for Log Gap, False for Raw Best Value
USE_LOG_GAP = True

GLOBAL_MINIMUM = {
    'branin2': 0.397887,
    'hartmann3': -3.86278,
    'hartmann6': -3.32237, 
    'griewank5': 0, 'griewank50': 0, 'griewank10': 0, 'griewank20': 0,
    'levy8': 0, 'levy10': 0, 'levy30': 0, 'levy20': 0,
    'exp5': 0, 'exp10': 0,
    'rosen14': 0, 'rosen20': 0, 'rosen10': 0, 
    'ackley5': 0, 'ackley10': 0, 'ackley20': 0,
    'rastrigin30':0, 'rastrigin5':0, 'rastrigin15':0,
    'weierstrass3': 0.01, 
    'robot4': None, 'robot3': None, 'portfolio5': None,
}

NEGATE = ['portfolio5', 'SVM3', 'XGBoost9', 'LightGBM16']

PARAMS_DICT = {
    # Baselines 
    "rbf":    {"label": "RBF",  "marker": "x", "linestyle": ":", "color": "#1f77b4", "lw": 1.8},
    "ma52":   {"label": "MA52", "marker": "x", "linestyle": ":", "color": "#33AE40", "lw": 1.8},
    "sinc":   {"label": "Sinc", "marker": "+", "linestyle": ":", "color": "#8c564b", "lw": 1.8},
    "ada":    {"label": "Ada",  "marker": "D", "linestyle": ":", "color": "#7f7f7f", "lw": 1.8},
    "rq":     {"label": "RQ",   "marker": "*", "linestyle": ":", "color": "#9467bd", "lw": 1.8},
    "sdk":    {"label": "SDK",  "marker": "+", "linestyle": "--","color": "#ccbc55", "lw": 1.8},
    "abo":    {"label": "ABO",  "marker": "1", "linestyle": ":", "color": "#2ca02c", "lw": 1.8},

    #Proposed Methods
    "learnable_alpha_q1": {
        "label": "ALAS(Coupled)", "marker": "o", "linestyle": "--", "color": "#d62728", "lw": 2.5,
    },
    "additive_stable_q1": {
        "label": "ALAS(Additive)", "marker": "D", "linestyle": "--", "color": "#dd6a30", "lw": 2.5,
    },
}

PAPER_TASKS = [
    # Row 1: Low Dim 
    ('hartmann3', '(b) Hartmann-3d'),
    ('weierstrass3', '(a) Weierstrass-3d'),
    ('robot4', '(c) Robot-4d'),
    
    # Row 2: Mid Dim
    ('portfolio5', '(d) Portfolio-5d'),
    ('exp5', '(e) Exponential-5d'),
    ('hartmann6', '(f) Hartmann-6d'),
    
    # Row 3: High Dim 
    ('rosen10', '(g) Rosenbrock-10d'), 
    ('levy20', '(h) Levy-20d'),
    ('rastrigin30', '(i) Rastrigin-30d'),
]

BAR_DICT_DEFAULT = {"beta": 1.0, "errorevery": 1} 

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_filename(filename):
    filename_no_ext = filename.replace(".pkl", "")
    parts = filename_no_ext.split('_')
    
    if len(parts) >= 3:
        function_name = parts[0]
        acq_function = parts[-1]
        kernel_identifier = "_".join(parts[1:-1])
        return function_name, kernel_identifier, acq_function
    return None, None, None

def get_plot_data(data_dir, target_func, target_acq):
    results = defaultdict(list)
    if not os.path.exists(data_dir):
        return results

    for filename in os.listdir(data_dir):
        if not filename.endswith(".pkl"):
            continue
        try:
            func, kernel, acq = parse_filename(filename)
            if func != target_func or acq != target_acq:
                continue
            
            if kernel not in PARAMS_DICT:
                continue

            file_path = os.path.join(data_dir, filename)
            data = load_pickle(file_path)
            valid_runs = [run for run in data if isinstance(run, list) and len(run) > 0]
            if valid_runs:
                results[kernel].extend(valid_runs)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return results

def process_runs(runs, func_name):
    min_len = min(len(r) for r in runs)
    if min_len == 0:
        return None, None, None, None

    processed = []
    global_min = GLOBAL_MINIMUM.get(func_name)

    if func_name == 'portfolio5':
        target_metric = 'max' 
    elif func_name == 'robot4':
        target_metric = 'min'
    else:
        if USE_LOG_GAP:
            target_metric = 'gap'
        else:
            target_metric = 'min'

    should_negate = func_name in NEGATE

    for r in runs:
        traj = np.array(r[:min_len])
        best_so_far = np.minimum.accumulate(traj)
        
        if should_negate:
            best_so_far = -best_so_far
            
        if target_metric == "gap":
            if global_min is not None:
                gap = np.abs(best_so_far - global_min)
                val = np.log(np.maximum(gap, 1e-16)) 
            else:
                val = best_so_far
        else:
            val = best_so_far
            
        processed.append(val)

    processed_np = np.array(processed)
    mean = np.mean(processed_np, axis=0)
    std = np.std(processed_np, axis=0) / np.sqrt(len(runs)) 
    iterations = np.arange(1, min_len + 1)
    
    return iterations, mean, std, target_metric

# main plotting function
def plot_paper_grid(data_dir, acq_to_plot='ei'):
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    legend_handles = {}
    
    mode_str = "Log Optimality Gap" if USE_LOG_GAP else "Best Observed Value"
    print(f"--- Generating Reference-Style Grid Plot ({acq_to_plot.upper()}) | Mode: {mode_str} ---")

    for idx, (func_name, plot_title) in enumerate(PAPER_TASKS):
        ax = axes[idx]
        kernel_data = get_plot_data(data_dir, func_name, acq_to_plot)
        
        y_axis_label = "Value"
        current_y_type = "min" 

        if not kernel_data:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(plot_title, y=-0.15)
            continue

        sorted_kernels = [k for k in PARAMS_DICT.keys() if k in kernel_data]

        for k_id in sorted_kernels:
            runs = kernel_data[k_id]
            iters, mean, std, y_type = process_runs(runs, func_name)
            if iters is None: continue
            
            current_y_type = y_type 

            style = PARAMS_DICT.get(k_id)
            n_points = len(iters)
            ee = 1 if n_points < 20 else int(n_points / 12)
            
            ax.plot(
                iters, mean, 
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=style['lw'],
                markersize=5,       
                markevery=ee      
            )
            
            #ax.fill_between(iters, mean - std, mean + std, color=style['color'], alpha=0.15)

            if style['label'] not in legend_handles:
                legend_handles[style['label']] = Line2D([0], [0], 
                                                        color=style['color'], 
                                                        linestyle=style['linestyle'], 
                                                        marker=style['marker'], 
                                                        lw=2)
        if current_y_type == "gap":
            y_axis_label = "Log Optimality Gap"
        elif current_y_type == "max":
            y_axis_label = "Max Objective"
        elif current_y_type == "min":
            y_axis_label = "Min Objective" 

        ax.set_title(plot_title, fontsize=12, pad=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        if y_axis_label != "Value" or idx % 3 == 0:
            ax.set_ylabel(y_axis_label, fontsize=10)
        elif current_y_type in ['max']: 
            ax.set_ylabel(y_axis_label, fontsize=10)

        if idx >= 6: ax.set_xlabel("# Iterations", fontsize=10)

    for i in range(len(PAPER_TASKS), 9):
        fig.delaxes(axes[i])

    if legend_handles:
        fig.legend(handles=legend_handles.values(), 
                   labels=legend_handles.keys(), 
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.92), 
                   ncol=7, 
                   frameon=False,
                   fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    root_dir = os.path.dirname(os.path.dirname(data_dir))
    save_dir = os.path.join(root_dir, "paper_figures")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    suffix = "log_gap" if USE_LOG_GAP else "raw_val"
    save_filename = f"grid_errorbar_{acq_to_plot}_{suffix}.pdf"
    save_path = os.path.join(save_dir, save_filename)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)

    print(f"Saved PDF plot to: {save_path}")

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root_dir = os.path.dirname(current_script_dir)
    results_pkl_dir = os.path.join(project_root_dir, 'exp_res_best', 'pkl')

    ACQ_TO_PLOT = 'ucb' 

    if not os.path.isdir(results_pkl_dir):
        print(f"Error: Results directory not found: {results_pkl_dir}")
    else:
        plot_paper_grid(results_pkl_dir, acq_to_plot=ACQ_TO_PLOT)
