import torch

import argparse
import pickle
import warnings
import os
import random
import time
import numpy as np
import traceback

warnings.filterwarnings("ignore")

from test_functions import (
    Ackley,
    Branin,
    Exponential,
    Griewank,
    Hartmann3,
    Hartmann6,
    Hartmann24,
    Levy,
    Rastrigin,
    Rosenbrock,
    Weierstrass,
)
from utils.utils import get_next_points

parser = argparse.ArgumentParser('Run BayesOpt Experiments')
parser.add_argument('--function_name', type=str, default='branin2', help='objective function')
parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
parser.add_argument('--n_init', type=int, default=5, help='number of initial random points')
parser.add_argument('--kernel', type=str, default='learnable_alpha_stable', help='choice of kernel')
parser.add_argument('--acq', type=str, default='ei', help='choice of the acquisition function')
parser.add_argument('--n_mixture', type=int, default=7, help='number of mixtures')
parser.add_argument('--n_mixture1', type=int, default=2, help='number of Cauchy mixtures')
parser.add_argument('--n_mixture2', type=int, default=2, help='number of Gaussian mixtures')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0, 2.0], help='List of fixed alpha values')
parser.add_argument('--seed', type=int, default=1, help='number of seeds to run; uses seeds 0, ..., seed-1')
parser.add_argument('--verbose', action='store_true', help='print detailed model-fitting diagnostics')

args = parser.parse_args()
options = vars(args)
if args.verbose:
    print(options)
else:
    print(
        f"Running {args.function_name} | kernel={args.kernel} | acq={args.acq} | "
        f"n_init={args.n_init} | n_iter={args.n_iter} | seeds={args.seed}"
    )

current_dir = os.path.dirname(os.path.abspath(__file__))

if args.kernel == 'csm' or args.kernel == 'gsm':
    filename = f"{args.function_name}_{args.kernel}{args.n_mixture}_{args.acq}.pkl"
elif args.kernel == 'mix':
    filename = f"{args.function_name}_c{args.n_mixture1}g{args.n_mixture2}_{args.acq}.pkl"
elif args.kernel == 'mixstable_fixed_alpha':
    alpha_str = '_'.join([f"{a:.1f}".replace('.', 'p') for a in args.alphas])
    filename = f"{args.function_name}_mixstable_a{alpha_str}_{args.acq}.pkl"
elif args.kernel == 'learnable_alpha_stable':
    filename = f"{args.function_name}_learnable_alpha_q{args.n_mixture}_{args.acq}.pkl"
elif args.kernel == 'alas':
    filename = f"{args.function_name}_alas_q{args.n_mixture}_{args.acq}.pkl"
elif args.kernel == 'additive_stable':
    filename = f"{args.function_name}_additive_stable_q{args.n_mixture}_{args.acq}.pkl"
elif args.kernel == 'alas_sep':
    filename = f"{args.function_name}_alas_sep_q{args.n_mixture}_{args.acq}.pkl"
else:
    filename = f"{args.function_name}_{args.kernel}_{args.acq}.pkl"

save_dir = os.path.join(current_dir, 'exp_res_best', 'pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, filename)
print(f"Saving results to: {os.path.relpath(file_path, current_dir)}")

# --------------------------------------------------

seed_list = range(0, args.seed, 1)
best_all = []
alpha_history_all_seeds = {}
time_start = time.time()

for seed in seed_list:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed {seed + 1}/{len(seed_list)} (id={seed})")

    alpha_history_current_seed = []
    
    if args.function_name == 'branin2':
        func = Branin()
    elif args.function_name == 'hartmann3':
        func = Hartmann3()
    elif args.function_name == 'griewank5':
        func = Griewank(dim=5)
    elif args.function_name == 'hartmann6':
        func = Hartmann6()
    elif args.function_name == 'hartmann24':
        func = Hartmann24()
    elif args.function_name == 'exp5':
        func = Exponential(dim=5)
    elif args.function_name == 'exp10':
        func = Exponential(dim=10)
    elif args.function_name == 'weierstrass6':
        func = Weierstrass(dim=6)
    elif args.function_name == 'rastrigin15':
        func = Rastrigin(dim=15)
    elif args.function_name == 'rastrigin20':           
        func = Rastrigin(dim=20)
    elif args.function_name == 'rastrigin30':
        func = Rastrigin(dim=30)
    elif args.function_name == 'griewank10':
        func = Griewank(dim=10)
    elif args.function_name == 'griewank20':
        func = Griewank(dim=20)
    elif args.function_name == 'griewank50':
        func = Griewank(dim=50)
    elif args.function_name == 'rosen10':
        func = Rosenbrock(dim=10)
    elif args.function_name == 'ackley10':
        func = Ackley(dim=10)
    elif args.function_name == 'ackley20':
        func = Ackley(dim=20)
    elif args.function_name == 'rosen20':
        func = Rosenbrock(dim=20)
    elif args.function_name == 'levy10':
        func = Levy(dim=10)
    elif args.function_name == 'levy20':
        func = Levy(dim=20)
    elif args.function_name == 'levy30':
        func = Levy(dim=30)
    elif args.function_name == 'robot3':
        from test_functions import Robot3

        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot3(gpos[0][0], gpos[0][1])
    elif args.function_name == 'robot4':
        from test_functions import Robot4

        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot4(gpos[0][0], gpos[0][1])
    elif args.function_name == 'portfolio5':
        from test_functions import PortfolioSurrogate

        func = PortfolioSurrogate()
    elif args.function_name == 'XGBoost9':
        from test_functions import XGBoost_HPO

        func = XGBoost_HPO()
    elif args.function_name == 'XGBoost14':
        from test_functions import XGBoost_HPO_14D

        func = XGBoost_HPO_14D()
    elif args.function_name == 'LightGBM16':
        from test_functions import LightGBM_HPO

        func = LightGBM_HPO()
    elif args.function_name == 'SVM3':
        from test_functions import SVM_HPO

        func = SVM_HPO()
    elif args.function_name == 'weierstrass3':
        func = Weierstrass(dim=3)
    else:
        raise ValueError('Unrecognised problem %s' % args.function_name)

    d = func.dim
    lb = func.lb
    ub = func.ub
    bounds = torch.stack((lb, ub))

    init_x = torch.rand(args.n_init, d, dtype=torch.float32)
    init_x = bounds[0] + (bounds[1] - bounds[0]) * init_x
    if args.function_name == 'robot3' or args.function_name == 'robot4':
        init_x = torch.mean(init_x, dim=0, keepdim=True)
    init_y = func.eval(init_x)
    best_init_y = init_y.min().item()

    n_iterations = args.n_iter
    best_result = [best_init_y]

    kernel_to_pass = args.kernel
    if kernel_to_pass == 'mixstable_fixed_alpha':
        if args.alphas is None or len(args.alphas) == 0:
            raise ValueError("`--alphas` argument is required")

    try:
        for i in range(n_iterations):
            alphas_to_pass = args.alphas if args.kernel == 'mixstable_fixed_alpha' else None

            try:
                new_candidates,stats= get_next_points(
                    acq=args.acq,
                    kernel=kernel_to_pass,
                    n_mixture=args.n_mixture,
                    init_x=init_x,
                    init_y=init_y,
                    best_init_y=best_init_y,
                    bounds=bounds,
                    n_points=1,
                    n_mixture1=args.n_mixture1,
                    n_mixture2=args.n_mixture2,
                    alphas=alphas_to_pass,
                    return_stats=True,
                    verbose=args.verbose 
                )
                if stats: 
                    alpha_history_current_seed.append(stats)
                    if args.verbose and stats.get("alpha_bar") is not None:
                        print(f"    alpha_bar={stats['alpha_bar']:.4f}")
            except Exception as e:
                print(f"Error in get_next_points: {e}")
                traceback.print_exc()
                break 

            new_results = func.eval(new_candidates)

            init_x = torch.cat([init_x, new_candidates])
            init_y = torch.cat([init_y, new_results])
            
            current_y_val = new_results.item()
            best_init_y = init_y.min().item()
            best_result.append(best_init_y)
            
            improvement = best_result[-2] - best_init_y
            iter_msg = (
                f"  iter {i + 1:03d}/{n_iterations:03d} | "
                f"y={current_y_val:.6g} | best={best_init_y:.6g}"
            )
            if improvement > 0:
                iter_msg += f" | improvement={improvement:.6g}"
            print(iter_msg)

            if args.verbose:
                dist = torch.cdist(new_candidates, init_x[:-1])
                min_dist = dist.min().item()
                print(f"    nearest_dist={min_dist:.6g}")

    except Exception as e:
        print(f"Seed {seed} CRITICAL FAILURE: {e}")
        traceback.print_exc()
    
    finally:
        best_all.append(best_result)
        if alpha_history_current_seed:
            alpha_history_all_seeds[seed] = alpha_history_current_seed
        
        with open(file_path, 'wb') as f:
            pickle.dump(best_all, f)
        print(f"Seed {seed + 1}/{len(seed_list)} saved.")

if alpha_history_all_seeds:
    alpha_file_path = file_path.replace(".pkl", "_alpha_history.pkl")
    print(f"Saving alpha history to: {os.path.relpath(alpha_file_path, current_dir)}")
    with open(alpha_file_path, 'wb') as f:
        pickle.dump(alpha_history_all_seeds, f)

time_end = time.time()
running_time = (time_end - time_start)/max(1, len(seed_list))
print(f"Total running time: {time_end - time_start:.2f}s")
print(f"Average time per seed: {running_time:.2f}s")
