import copy
import math
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise


AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'


def geometric_median_update(points, alphas, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6):
    """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
    """
    alphas = np.asarray(alphas, dtype=points[0][0].dtype) / sum(alphas)
    median = weighted_average_oracle(points, alphas)
    num_oracle_calls = 1

    # logging
    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)
    if verbose:
        print('Starting Weiszfeld algorithm')
        print(log_entry)

    # start
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = np.asarray([alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, points)],
                             dtype=alphas.dtype)
        weights = weights / weights.sum()
        median = weighted_average_oracle(points, weights)
        num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i + 1, obj_val,
                     (prev_obj_val - obj_val) / obj_val,
                     l2dist(median, prev_median)]
        logs.append(log_entry)
        if verbose:
            print(log_entry)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
    return median, num_oracle_calls, logs


def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective."""
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])


def l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a list of nd-arrays"""
    return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])


def weighted_average_oracle(points, weights):
    """Computes weighted average of atoms with specified weights
    Args:
        points: list, whose weighted average we wish to calculate
            Each element is a list_of_np.ndarray
        weights: list of weights of the same length as atoms
    """
    tot_weights = np.sum(weights)
    weighted_updates = [np.zeros_like(v) for v in points[0]]

    for w, p in zip(weights, points):
        for j, weighted_val in enumerate(weighted_updates):
            weighted_val += (w / tot_weights) * p[j]

    return weighted_updates


def update(updates, aggregation=AGGR_GEO_MED, max_update_norm=None, maxiter=4):
    """Updates server model using given client updates.
    Args:
        updates: list of (num_samples, update), where num_samples is the
            number of training samples corresponding to the update, and update
            is a list of variable weights
        aggregation: Algorithm used for aggregation. Allowed values are:
            [ 'mean', 'geom_median']
        max_update_norm: Reject updates larger than this norm,
        maxiter: maximum number of calls to the Weiszfeld algorithm if using the geometric median
    """

    def accept_update(u):
        norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
        return not (np.isinf(norm) or np.isnan(norm))

    all_updates = updates
    updates = [u for u in updates if accept_update(u)]
    if len(updates) < len(all_updates):
        print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
    if len(updates) == 0:
        print('All individual updates rejected. Continuing without update')
        return 1, False

    points = [u[1] for u in updates]
    alphas = [u[0] for u in updates]
    if aggregation == AGGR_MEAN:
        weighted_updates = weighted_average_oracle(points, alphas)
        num_comm_rounds = 1
    elif aggregation == AGGR_GEO_MED:
        weighted_updates, num_comm_rounds, _ = geometric_median_update(points, alphas, maxiter=maxiter)
    else:
        raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

    update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])

    if max_update_norm is None or update_norm < max_update_norm:
        print(len(weighted_updates))
        updated = True
    else:
        print('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
        updated = False

    return weighted_updates, num_comm_rounds, updated


def trimmed_mean(w, trim_ratio=0.1):
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    cur_time = time.time()
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    print('model aggregation "trimmed mean" took {}s'.format(time.time() - cur_time))
    return w_med


def rfa(ws):
    updates_poi = []
    gradients = []
    for gradient in ws:
        gradients.append(np.array(orderdict_tolist(gradient)))
    for updat in gradients:
        updates_poi.append((5000, updat))
    global_model, upda, i = update(updates_poi)
    global_model = [i.item(0) for i in global_model]
    return list_todict(global_model)


"""
Util methods
"""


def list_todict(weight_list, shape=nn.Module):
    w_shape = shape.state_dict()
    weight_dict = dict(w_shape.items())
    start_index = 0
    for key in weight_dict.keys():
        key_size = len(torch.reshape(weight_dict[key], (-1,)).tolist())
        tmp = weight_list[start_index:start_index + key_size]
        weight_dict[key] = torch.reshape(torch.Tensor(tmp), weight_dict[key].size())
        start_index = start_index + key_size
    return (weight_dict)


def orderdict_tolist(w):
    weight_dict = dict(w.items())
    weight_list = []
    for key in weight_dict.keys():
        # print(key)
        # print(weight_dict[key].shape)
        weight_list = weight_list + torch.reshape(weight_dict[key], (-1,)).tolist()
    return weight_list


def orderdict_tolist_adapt(w, FMnist=False):
    weight_dict = dict(w.items())
    weight_list = []

    if FMnist:
        layer = "fc1"
        size = 512

    else:
        layer = "fc2"
        size = 50

    for key in weight_dict.keys():
        if key == (layer + ".weight"):
            # print(key)
            # print(weight_dict[key].shape)
            target = torch.zeros(11, size, dtype=torch.float)
            source = weight_dict[key]
            target[:10, :] = source
            weight_list = weight_list + torch.reshape(target, (-1,)).tolist()
        else:
            if key == (layer + ".bias"):
                target = torch.zeros(11, dtype=torch.long)
                source = weight_dict[key]
                target[:10] = source
                weight_list = weight_list + torch.reshape(target, (-1,)).tolist()
            else:
                weight_list = weight_list + torch.reshape(weight_dict[key], (-1,)).tolist()

    return weight_list


def krum(ws, args, f=0, m=None, **kwargs):
    gradients = []
    for gradient in ws:
        gradients.append(torch.Tensor(orderdict_tolist(gradient)))
    n = len(gradients)
    # Defaults
    if m is None:
        m = n - f - 2
    # Compute all pairwise distances
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        dist = gradients[x].sub(gradients[y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist
    # Compute the scores
    scores = list()
    for i in range(n):
        # Collect the distances
        grad_dists = list()
        for j in range(i):
            grad_dists.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
        for j in range(i + 1, n):
            grad_dists.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
        # Select the n - f - 1 smallest distances
        grad_dists.sort()
        scores.append((sum(grad_dists[:n - f - 1]), gradients[i]))
    # Compute the average of the selected gradients
    scores.sort(key=lambda x: x[0])
    tmp = sum(grad for _, grad in scores[:m]).div_(m)
    return list_todict(tmp, args)


"""
Norm clipping defence mechanism
"""


def normBound(ws, args):
    gradients = []
    clipped_gradients = []
    for gradient in ws:
        gradients.append(torch.Tensor(orderdict_tolist(gradient)))
    n = len(gradients)
    # Compute all norms and clip them
    for grad in gradients:
        print("Before clipping : ", grad.norm().item())
        clipped_gradients.append(grad / max(1, grad.norm().item() / args.bound))
        # clipped_gradients.append(grad / max(1, torch.norm(grad).item()/args.bound))

    for grad in clipped_gradients:
        print("After clipping : ", grad.norm().item())

    tmp = sum(grad for grad in clipped_gradients).div_(n)
    return list_todict(tmp, args)


def FLaggregate(ws, args):
    gradients = []
    clipped_gradients = []
    for gradient in ws:
        gradients.append(torch.Tensor(orderdict_tolist(gradient)))
    n = len(gradients)
    # Compute all norms and clip them

    tmp = sum(grad for grad in gradients).div_(n)
    return list_todict(tmp, args)


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

def average_weights(w, args):
    """
    Returns the average of the weights.
    """


    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg, [w_avg for x in range(len(w))]

