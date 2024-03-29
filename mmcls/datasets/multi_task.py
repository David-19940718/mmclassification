# Copyright (c) Jack.Wang. All rights reserved.
import warnings
from typing import List
from itertools import chain

import numpy as np

from mmcls.core import average_performance, mAP
from .base_dataset import BaseDataset


class MultiTaskDataset(BaseDataset):
    """Multi-task Dataset."""

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image categories of specified index.
        """
        gt_labels = self.data_infos[idx]['gt_label']
        cat_ids = np.where(gt_labels == 1)[0].tolist()
        return cat_ids

    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 logger=None,
                 **deprecated_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.

        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'thr': 0.5}

        if deprecated_kwargs != {}:
            warnings.warn('Option arguments for metrics has been changed to '
                          '`metric_options`.')
            metric_options = {**deprecated_kwargs}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        eval_results = {}

        # 拆成num_task个
        num_tasks = metric_options['num_tasks']
        tmp = list(chain.from_iterable(results))
        tmp = [tmp[i: i+num_tasks] for i in range(0, len(tmp), num_tasks)]
        results = []
        # loop i-th samples
        for sample in tmp:
            # get index
            arr = np.zeros(num_tasks)
            for j in range(num_tasks):
                arr[j] = np.argmax(sample)
            results.append(arr)

        results, gt_labels = np.array(results), np.array(self.get_gt_labels())

        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            performance_values = average_performance(results, gt_labels, thr=metric_options['thr'])
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results
