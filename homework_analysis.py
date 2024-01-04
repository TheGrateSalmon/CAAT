import argparse
from collections import Counter
import functools as ft
import itertools as it
import more_itertools as mit
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt
# plt.style.use('classic')


BASE_SEQ = {4: [[1,2,3,4], [1,2,2,2], [1,2,3,3], [1,2,2,4], [1,1,3,3], [1,1,3,4], [1,1,1,4], [1,1,1,1]],
            5: [[1,2,3,4,5], [1,1,1,1,1], [1,2,2,2,2], [1,1,1,1,5], [1,2,2,2,5], [1,2,3,3,3], [1,1,1,4,5], [1,1,3,3,5], 
                [1,2,2,4,4], [1,2,2,4,5], [1,2,3,3,5], [1,2,3,4,4], [1,1,1,4,4], [1,1,3,3,3], [1,1,3,4,5], [1,1,3,4,4]],
            6: [[1,2,3,4,5,6], [1,1,1,1,1,1], [1,1,1,1,1,6], [1,2,2,2,2,2], [1,1,1,1,5,5], [1,1,3,3,3,3], [1,1,1,4,4,4], [1,1,1,4,4,6], 
                [1,1,1,4,5,5], [1,2,2,2,5,5], [1,1,3,3,3,6], [1,1,3,4,4,4], [1,2,2,4,4,4], [1,1,1,1,5,6], [1,2,3,3,3,3], [1,2,2,2,2,6], 
                [1,1,3,3,5,5], [1,1,1,4,5,6], [1,2,2,2,5,6], [1,2,3,3,3,6], [1,2,3,4,4,4], [1,1,3,3,5,6], [1,1,3,4,5,5], [1,1,3,4,4,6], 
                [1,2,2,4,5,5], [1,2,3,3,5,5], [1,2,2,4,4,6], [1,1,3,4,5,6], [1,2,3,4,5,5], [1,2,2,4,5,6], [1,2,3,3,5,6], [1,2,3,4,4,6]],
            7: [[1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 7], [1, 2, 2, 2, 2, 2, 2], 
                [1, 1, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 6, 6], [1, 1, 1, 1, 5, 5, 5], [1, 1, 1, 4, 4, 4, 4], 
                [1, 1, 1, 1, 1, 6, 7], [1, 2, 2, 2, 2, 2, 7], [1, 2, 3, 3, 3, 3, 3], [1, 1, 1, 1, 5, 5, 7], 
                [1, 2, 2, 2, 2, 6, 6], [1, 1, 3, 3, 3, 3, 7], [1, 2, 2, 4, 4, 4, 4], [1, 1, 3, 4, 4, 4, 4], 
                [1, 1, 1, 1, 5, 6, 6], [1, 1, 1, 4, 4, 4, 7], [1, 1, 1, 4, 5, 5, 5], [1, 2, 2, 2, 5, 5, 5], 
                [1, 1, 1, 4, 4, 6, 6], [1, 1, 3, 3, 3, 6, 6], [1, 1, 3, 3, 5, 5, 5], [1, 1, 1, 1, 5, 6, 7], 
                [1, 2, 2, 2, 2, 6, 7], [1, 2, 3, 3, 3, 3, 7], [1, 2, 3, 4, 4, 4, 4], [1, 1, 1, 4, 4, 6, 7], 
                [1, 1, 1, 4, 5, 5, 7], [1, 1, 1, 4, 5, 6, 6], [1, 1, 3, 3, 3, 6, 7], [1, 1, 3, 4, 4, 4, 7], 
                [1, 1, 3, 4, 5, 5, 5], [1, 2, 2, 2, 5, 5, 7], [1, 2, 2, 4, 4, 4, 7], [1, 2, 2, 2, 5, 6, 6], 
                [1, 2, 2, 4, 5, 5, 5], [1, 2, 3, 3, 5, 5, 5], [1, 2, 3, 3, 3, 6, 6], [1, 1, 3, 3, 5, 5, 7], 
                [1, 1, 3, 3, 5, 6, 6], [1, 1, 3, 4, 4, 6, 6], [1, 2, 2, 4, 4, 6, 6], [1, 1, 1, 4, 5, 6, 7], 
                [1, 2, 2, 2, 5, 6, 7], [1, 2, 3, 3, 3, 6, 7], [1, 2, 3, 4, 4, 4, 7], [1, 2, 3, 4, 5, 5, 5], 
                [1, 1, 3, 3, 5, 6, 7], [1, 1, 3, 4, 4, 6, 7], [1, 1, 3, 4, 5, 5, 7], [1, 1, 3, 4, 5, 6, 6], 
                [1, 2, 3, 4, 4, 6, 6], [1, 2, 3, 3, 4, 6, 6], [1, 2, 2, 4, 5, 6, 6], [1, 2, 2, 4, 4, 6, 7], 
                [1, 2, 3, 3, 5, 5, 7], [1, 2, 2, 4, 5, 5, 7], [1, 1, 3, 4, 5, 6, 7], [1, 2, 2, 4, 5, 6, 7], 
                [1, 2, 3, 3, 5, 6, 7], [1, 2, 3, 4, 4, 6, 7], [1, 2, 3, 4, 5, 5, 7], [1, 2, 3, 4, 5, 6, 6]],
            8: [[1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 8], [1, 2, 2, 2, 2, 2, 2, 2], 
                [1, 1, 1, 1, 1, 1, 7, 7], [1, 1, 3, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 6, 6, 6], [1, 1, 1, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 5, 5, 5, 5], [1, 1, 1, 1, 1, 1, 7, 8], [1, 2, 2, 2, 2, 2, 2, 8], [1, 2, 3, 3, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 6, 6, 8], [1, 1, 1, 1, 1, 6, 7, 7], [1, 2, 2, 2, 2, 2, 7, 7], [1, 2, 2, 4, 4, 4, 4, 4],
                [1, 1, 3, 3, 3, 3, 3, 8], [1, 1, 2, 4, 4, 4, 4, 4], [1, 1, 1, 1, 5, 5, 5, 8], [1, 1, 1, 1, 5, 6, 6, 6],
                [1, 1, 1, 4, 4, 4, 4, 8], [1, 1, 1, 4, 5, 5, 5, 5], [1, 2, 2, 2, 2, 6, 6, 6], [1, 2, 2, 2, 5, 5, 5, 5],
                [1, 1, 1, 1, 5, 5, 7, 7], [1, 1, 3, 3, 3, 3, 7, 7], [1, 1, 3, 3, 5, 5, 5, 5], [1, 1, 1, 4, 4, 4, 7, 7],
                [1, 1, 1, 4, 4, 6, 6, 6], [1, 1, 3, 3, 3, 6, 6, 6], [1, 1, 1, 1, 1, 6, 7, 8], [1, 2, 2, 2, 2, 2, 7, 8],
                [1, 2, 3, 3, 3, 3, 3, 8], [1, 2, 3, 4, 4, 4, 4, 4], [1, 1, 1, 1, 5, 5, 7, 8], [1, 1, 1, 1, 5, 6, 6, 8],
                [1, 1, 1, 1, 5, 6, 7, 7], [1, 1, 3, 3, 3, 3, 7, 8], [1, 1, 3, 4, 4, 4, 4, 8], [1, 1, 3, 4, 5, 5, 5, 5],
                [1, 2, 3, 3, 3, 3, 7, 7], [1, 2, 3, 3, 5, 5, 5, 5], [1, 2, 2, 2, 2, 6, 6, 8], [1, 2, 2, 4, 4, 4, 4, 8],
                [1, 2, 2, 4, 5, 5, 5, 5], [1, 2, 2, 2, 2, 6, 7, 7], [1, 1, 1, 4, 4, 4, 7, 8], [1, 1, 1, 4, 5, 5, 5, 8],
                [1, 1, 1, 4, 5, 6, 6, 6], [1, 2, 2, 2, 5, 5, 5, 8], [1, 2, 3, 3, 3, 6, 6, 6], [1, 2, 2, 2, 5, 6, 6, 6],
                [1, 1, 1, 4, 4, 6, 6, 8], [1, 1, 1, 4, 4, 6, 7, 7], [1, 1, 1, 4, 5, 5, 7, 7], [1, 1, 3, 3, 3, 6, 6, 8],
                [1, 1, 3, 3, 3, 6, 7, 7], [1, 1, 3, 4, 4, 4, 7, 7], [1, 1, 3, 4, 4, 6, 6, 6], [1, 1, 3, 3, 5, 6, 6, 6],
                [1, 1, 3, 3, 5, 5, 5, 8], [1, 2, 2, 2, 5, 5, 7, 7], [1, 2, 2, 4, 4, 4, 7, 7], [1, 2, 2, 4, 4, 6, 6, 6],
                [1, 1, 3, 3, 5, 5, 7, 7], [1, 1, 1, 1, 5, 6, 7, 8], [1, 2, 2, 2, 2, 6, 7, 8], [1, 2, 3, 3, 3, 3, 7, 8],
                [1, 2, 3, 4, 4, 4, 4, 8], [1, 2, 3, 4, 5, 5, 5, 5], [1, 1, 1, 4, 4, 6, 7, 8], [1, 1, 1, 4, 5, 5, 7, 8],
                [1, 1, 1, 4, 5, 6, 6, 8], [1, 1, 1, 4, 5, 6, 7, 7], [1, 1, 3, 3, 3, 6, 7, 8], [1, 1, 3, 4, 4, 4, 7, 8],
                [1, 1, 3, 4, 5, 5, 5, 8], [1, 1, 3, 4, 5, 6, 6, 6], [1, 2, 2, 2, 5, 5, 7, 8], [1, 2, 2, 2, 5, 6, 6, 8],
                [1, 2, 2, 2, 5, 6, 7, 7], [1, 2, 2, 4, 4, 4, 7, 8], [1, 2, 2, 4, 5, 5, 5, 8], [1, 2, 2, 4, 5, 6, 6, 6],
                [1, 2, 3, 3, 3, 6, 6, 8], [1, 2, 3, 3, 3, 6, 7, 7], [1, 2, 3, 3, 5, 5, 5, 8], [1, 2, 3, 3, 5, 6, 6, 6],
                [1, 2, 3, 4, 4, 4, 7, 7], [1, 2, 3, 4, 4, 6, 6, 6], [1, 1, 3, 3, 5, 5, 7, 8], [1, 1, 3, 3, 5, 6, 6, 8],
                [1, 1, 3, 3, 5, 6, 7, 7], [1, 1, 3, 4, 4, 6, 6, 8], [1, 1, 3, 4, 4, 6, 7, 7], [1, 1, 3, 4, 5, 5, 7, 7],
                [1, 2, 3, 3, 5, 5, 7, 7], [1, 2, 2, 4, 5, 5, 7, 7], [1, 2, 2, 4, 4, 6, 6, 8], [1, 2, 2, 4, 4, 6, 7, 7],
                [1, 1, 1, 4, 5, 6, 7, 8], [1, 2, 2, 2, 5, 6, 7, 8], [1, 2, 3, 3, 3, 6, 7, 8], [1, 2, 3, 4, 4, 4, 7, 8],
                [1, 2, 3, 4, 5, 5, 5, 8], [1, 2, 3, 4, 5, 6, 6, 6], [1, 1, 3, 3, 5, 6, 7, 8], [1, 1, 3, 4, 4, 6, 7, 8],
                [1, 1, 3, 4, 5, 5, 7, 8], [1, 1, 3, 4, 5, 6, 6, 8], [1, 1, 3, 4, 5, 6, 7, 7], [1, 2, 2, 4, 4, 6, 7, 8],
                [1, 2, 2, 4, 5, 5, 7, 8], [1, 2, 2, 4, 5, 6, 6, 8], [1, 2, 2, 4, 5, 6, 7, 7], [1, 2, 3, 3, 5, 5, 7, 8],
                [1, 2, 3, 3, 5, 6, 6, 8], [1, 2, 3, 3, 5, 6, 7, 7], [1, 2, 3, 4, 4, 6, 6, 8], [1, 2, 3, 4, 4, 6, 7, 7],
                [1, 2, 3, 4, 5, 5, 7, 7], [1, 1, 3, 4, 5, 6, 7, 8], [1, 2, 2, 4, 5, 6, 7, 8], [1, 2, 3, 3, 5, 6, 7, 8],
                [1, 2, 3, 4, 4, 6, 7, 8], [1, 2, 3, 4, 5, 5, 7, 8], [1, 2, 3, 4, 5, 6, 6, 8], [1, 2, 3, 4, 5, 6, 7, 7]]}


def weak_ord_dist(wo1: np.ndarray, wo2: np.ndarray):
    return np.linalg.norm(wo1-wo2, ord=2)


def weak_order_list_dist(weak_orders):
    weak_orders1, weak_orders2 = weak_orders
    counts = Counter()
    for i, wo1 in enumerate(weak_orders1):
        for j, wo2 in enumerate(weak_orders2):
            counts.update([round(weak_ord_dist(np.array(wo1), np.array(wo2)), 6)])
    return counts


def compute_weak_order_dists(weak_orders: List[List[Tuple]]) -> Counter:
    """Computes the distances between all weak orders of a given length.

    Parameters
    ----------
    weak_orders : List[List[Tuple]]
        A collection of lists, where each list consists of all permutations of a weak order
    
    Returns
    -------
    counts : Counter
        A counter of all the distances between weak orders.
    """
    num_dists = (len(weak_orders) * (len(weak_orders)+1))//2
    p = mp.Pool()
    counts = p.imap(weak_order_list_dist, it.combinations_with_replacement(weak_orders, r=2), chunksize=num_dists//os.cpu_count())
    counts = ft.reduce(lambda x, y: x+y, counts)
    p.close()
    p.join()
    return counts


def weighted_quantiles(values, quantiles, sample_weights=None) -> np.ndarray:
    """Computes quantiles of weighted data.

    See https://stackoverflow.com/a/29677616

        
    Parameters
    ----------
    values : array-like
        numpy array of data
    quantiles : array-like
        numpy array of quantiles to compute
    sample_weights : array-like
        numpy array of the same length as `values`, default is None

    Returns
    -------
        numpy array of computed quantiles based on weighted samples
    """
    if sample_weights is None:
        sample_weights = np.ones(len(values))
    values, quantiles, sample_weights = np.array(values), np.array(quantiles), np.array(sample_weights)

    # sort values
    sorter = np.argsort(values)
    values, sample_weights = values[sorter], sample_weights[sorter]
    
    # compute weighted quantiles
    weighted_quantiles = np.cumsum(sample_weights) - 0.5 * sample_weights
    weighted_quantiles /= np.sum(sample_weights)

    return np.interp(quantiles, weighted_quantiles, values)


def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='sequence_length', action='store', required=True, 
                        type=int, default=4, choices=list(BASE_SEQ.keys()),
                        help='Length of sequence')
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # make weak orders
    # n = int(input('Length of sequence: '))
    n = args.sequence_length
    weak_orders = [list(mit.distinct_permutations(seq)) for seq in BASE_SEQ[n]]
    
    # compute distances between weak orders
    start_time = time.time()
    counts = compute_weak_order_dists(weak_orders)
    end_time = time.time()

    # turn counts into relative frequencies        
    total = sum(counts.values())
    rel_freq = {dist: count/total for (dist, count) in counts.items()}
    
    # compute quartiles
    quartiles = weighted_quantiles(list(counts.keys()), [0.25, 0.5, 0.75], sample_weights=list(counts.values()))
    print(f'Quartiles: {quartiles}')

    # create the distance-frequency plot
    xs, ys = list(rel_freq.keys()), list(rel_freq.values())
    plt.title(f'Frequency of Distances Between Ranking Sequences ($n={n}$)')
    plt.xlabel('Distance')
    plt.ylabel('Relative Frequency')
    plt.scatter(xs, ys, alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # quartile lines
    x_margin, y_margin = 0.1*(max(xs) - min(xs)), 0.1*(max(ys) - min(ys))
    plt.vlines(quartiles, ymin=min(ys)-y_margin, ymax=max(ys)+y_margin, linestyles='dashed', alpha=0.5)
    # change ytick labels to percentage
    ylocs, ylabels = plt.yticks()
    plt.yticks(ylocs, [f'{str(round(100*yloc,2))}%' for yloc in ylocs])
    plt.savefig(base_dir / 'figures' / f'weak_orders_{n}')

    # find the most common distance and sequences that yield it
    most_common_dist = counts.most_common(1)[0][0] 
    print(f'Most common distance: {most_common_dist}')

    # find the maximum distance and the sequences that yield it
    max_dist = max(counts.keys())
    print(f'Maximum distance: {max_dist}')

    with open(base_dir / 'weak_orders_stats.txt', 'a') as f:
        f.write(f'n={n}\n------------------------------\n')
        f.write(f'Quartiles: {quartiles.tolist()}\n')
        f.write(f'Most common distance: {most_common_dist}\n')
        f.write(f'Maximum distance: {max_dist}\n')
        f.write(f'Time to compute: {round((end_time-start_time)/60, 2)} minutes\n')
        f.write('\n'*3)


if __name__ == '__main__':
    main()