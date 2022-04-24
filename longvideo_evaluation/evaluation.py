import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from dataset import LongVideoDataset
from results import Results


class Evaluation(object):
    def __init__(self, root_folder, task, sequences='all', filelist=None):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.root_folder = root_folder
        self.task = task
        self.dataset = LongVideoDataset(root_folder=root_folder, task=task, sequences=sequences, filelist=filelist)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'NF' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'NF' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'NF' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    def evaluate(self, res_path, metric=('J', 'NF'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'NF' not in metric:
            raise ValueError('Metric possible values are J for IoU or NF for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'NF' in metric:
            metrics_res['NF'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:, :, :], all_masks_id[1:]
            all_res_masks = results.read_masks(seq, all_masks_id)
            if self.task == 'unsupervised':
                # j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, metric)
                pass
            elif self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    print('J', j_metrics_res[ii])
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'NF' in metric:
                    print('F', f_metrics_res[ii])
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['NF']["M"].append(FM)
                    metrics_res['NF']["R"].append(FR)
                    metrics_res['NF']["D"].append(FD)
                    metrics_res['NF']["M_per_object"][seq_name] = FM
                plt.figure()
                plt.plot(j_metrics_res[ii])
                plt.plot(f_metrics_res[ii])
                plt.show()

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
