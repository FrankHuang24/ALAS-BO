"""
Analyze learned stability index (alpha) from saved BO runs.

This script reads per-seed alpha snapshots (saved by utils.py) and summarizes
the final learned alpha (mean±std across seeds) and the overall alpha range
(10–90% quantiles across all iterations and seeds). It also prints a LaTeX table.
"""
import os
import pickle
import numpy as np


TARGET_KERNEL = 'additive_stable_q1' 
TASKS_CONFIG = [
    ('branin2',      'Smooth'),
    ('hartmann3',    'Smooth'),
    ('weierstrass_torture', 'Irregular'),
    ('exp5',         'Smooth'),
    ('levy20',       'Irregular'),
    ('hartmann6',    'Smooth'),
    ('rosen10',      'Smooth'),
    ('portfolio5',   'Irregular'), 
    ('robot4',       'Irregular'),
    ('rastrigin30',  'Irregular'),
]

RESULTS_PKL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_res_best', 'pkl')

#key process function
def process_alpha_history(data: dict) -> dict:
    final_alphas = []
    all_alphas = []

    if not data: return None

    for seed, history in data.items():
        if not history: continue
        last_stats = history[-1]
    
        if last_stats and 'alpha_bar' in last_stats:
            val = last_stats['alpha_bar']
            final_alphas.append(val)
        
        for stats in history:
            if stats and 'alpha_bar' in stats:
                all_alphas.append(stats['alpha_bar'])

    if not final_alphas: return None

    vals = np.array(all_alphas, dtype=float)
    
    stats = {
        'mean': np.mean(final_alphas),
        'std': np.std(final_alphas, ddof=1) if len(final_alphas) > 1 else 0.0,
        'lo': np.quantile(vals, 0.10),
        'hi': np.quantile(vals, 0.90), 
    }
    return stats

def main():
    print(f"--- Alpha History Analysis for Kernel: {TARGET_KERNEL} ---\n")
    
    table_rows = []

    for func_name, func_type in TASKS_CONFIG:
        filename = f"{func_name}_{TARGET_KERNEL}_ei_alpha_history.pkl"
        file_path = os.path.join(RESULTS_PKL_DIR, filename)
        
        row_data = {
            "name": func_name,
            "type": func_type,
            "res": "N/A", 
            "rng": "N/A" 
        }

        if not os.path.exists(file_path):
            print(f"[Missing] {filename}")
        else:
            try:
                with open(file_path, 'rb') as f:
                    history_data = pickle.load(f)
                
                stats = process_alpha_history(history_data)
                
                if stats:
                    print(f"[OK] {filename} -> {stats['mean']:.2f}")
                    row_data["res"] = f"{stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
                    row_data["rng"] = f"[{stats['lo']:.2f}, {stats['hi']:.2f}]"
                else:
                    print(f"[Empty] {filename}")
                    row_data["res"] = "Error"
            except Exception as e:
                print(f"[Error] Reading {filename}: {e}")

        table_rows.append(row_data)
    print("\n\n% --- LaTeX Code Start ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Learned stability index $\\alpha$ for {TARGET_KERNEL.replace('_', '-')} across benchmarks.}}")
    print("\\label{tab:alpha_learning}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Task (d) & Type & $\\alpha_\\text{final}$ (mean $\\pm$ std) & $\\alpha_\\text{range}$ (10--90\\%) \\\\")
    print("\\midrule")
    
    for row in table_rows:
        raw_name = row['name']
        dim = ''.join(filter(str.isdigit, raw_name))
        name_part = ''.join(filter(str.isalpha, raw_name)).capitalize()
        if "_" in raw_name:
            parts = raw_name.split('_')
            name_part = parts[0].capitalize() 
        
        if not dim and 'portfolio' in raw_name: dim = '5'
        if not dim and 'robot' in raw_name: dim = '4'

        print(f"{name_part}({dim}) & {row['type']} & {row['res']} & {row['rng']} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% --- LaTeX Code End ---")

if __name__ == '__main__':
    main()