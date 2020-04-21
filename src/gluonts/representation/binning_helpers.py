# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np

from gluonts.model.common import Tensor

# Workaround needed due to a known issue with np.quantile(inp, quant) returning unsorted values.
# We fix this by ensuring that the obtained bin_centers are monotonically increasing.
# Tracked in the following issues:
# - https://github.com/numpy/numpy/issues/14685
# - https://github.com/numpy/numpy/issues/12282
def ensure_binning_monotonicity(bin_centers: np.ndarray):
    for i in range(1, len(bin_centers)):
        if bin_centers[i] < bin_centers[i - 1]:
            bin_centers[i] = bin_centers[i - 1]
    return bin_centers


def bin_edges_from_bin_centers(bin_centers: np.ndarray):
    lower_edge = -np.inf
    upper_edge = np.inf
    bin_edges = np.concatenate(
        [
            [lower_edge],
            (bin_centers[1:] + bin_centers[:-1]) / 2.0,
            [upper_edge],
        ]
    )
    return bin_edges


def mxnet_bin_edges_from_bin_centers(F, bin_centers: Tensor):
    lower_edge = -np.inf
    upper_edge = np.inf
    bin_edges = F.concatenate(
        [
            F.full(1, lower_edge),
            (bin_centers[1:] + bin_centers[:-1]) / 2.0,
            F.full(1, upper_edge),
        ]
    )
    return bin_edges


def mxnet_quantile(F, x: Tensor, quantile_levels: Tensor):
    # print("-----")
    # print(x)
    x_sort = F.sort(x)
    quantile_ind = (quantile_levels * (len(x))).astype(int)
    quantiles = F.take(x_sort, quantile_ind)
    # print(x_sort)
    # print(quantile_levels)
    # print(quantile_ind)
    # print(quantiles)
    return quantiles


def mxnet_digitize(F, x: Tensor, bins: Tensor, num_bins: int):
    bins = F.repeat(bins.expand_dims(axis=0), repeats=len(x), axis=0)
    # bins_broad = F.broadcast_like(bins.expand_dims(axis=0), x)
    # print(bins)
    # print(bins_rep)
    # print(bins_broad)

    x = x.expand_dims(axis=-1)

    # exit(0)
    bins = bins.expand_dims(axis=-1).swapaxes(1, 2)

    left_edges = bins.slice_axis(axis=-1, begin=0, end=-1)
    right_edges = bins.slice_axis(axis=-1, begin=1, end=None)

    lesser = F.broadcast_lesser(x, right_edges)
    lesser_eq = F.broadcast_lesser_equal(left_edges, x)
    mask = F.broadcast_mul(lesser_eq, lesser)

    bin_slices = F.arange(num_bins) * F.ones_like(x)
    binning = F.broadcast_mul(bin_slices, mask).sum(axis=-1)

    return binning
