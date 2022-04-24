import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from evaluation import Evaluation

"""
Data structure tree
-longvideo
    -Annotations
    -JPEGImages
    -mask_palette.png
    -val.txt
"""

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/Ship01/Dataset/VOS/longvideo',
                    help='Path to the longvideo folder.',)
parser.add_argument('--method', type=str,
                    help='Method name.')
parser.add_argument('--results_path', type=str, required=True,
                    help='Path to the folder containing the sequences folders')
parser.add_argument('--filelist', type=str, default='val.txt',
                    help='Method name.')
parser.add_argument('--update', action='store_true',
                    help='Recompute the performance results.' )
args, _ = parser.parse_known_args()
args.task = 'semi-supervised'
csv_name_global = 'global_results.csv'
csv_name_per_sequence = 'per-sequence_results.csv'
# method_results_path = os.path.join(default_results_path, args.method + '_segs')
method_results_path = args.results_path
print('Evaluate', args.method)

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(method_results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(method_results_path, csv_name_per_sequence)
if not args.update and os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    dataset_eval = Evaluation(root_folder=args.path, task=args.task, filelist=args.filelist)
    metrics_res = dataset_eval.evaluate(method_results_path)
    J, F = metrics_res['J'], metrics_res['NF']

    # Generate dataframe for the general results
    g_measures = ['J&NF-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'NF-Mean', 'NF-Recall', 'NF-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'NF-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"------------------------ Global results for {args.method} ------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.method} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
print('')
