import os
import sys

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

import libcp
from src.ply_c import libply_c


def zhang(x, lam):
    """ Zhang's loss function.
    :param x: input
    :param lam: lambda
    """
    return torch.clamp(-lam * x + lam, min=0)


def compute_weight_loss(embeddings, edg_source, edg_target, is_transition, diff, xyz=0):
    transition_factor = 5
    k_nn_adj = 5
    pred_components, pred_in_component = compute_partition(embeddings, edg_source, edg_target, diff, xyz)

    weights_loss = compute_weights_XPART(pred_in_component, edg_source,
                                         edg_target, is_transition.cpu().numpy(),
                                         transition_factor * 2 * k_nn_adj)

    weights_loss = torch.from_numpy(weights_loss).cuda()

    return weights_loss, pred_components, pred_in_component


def compute_weights_XPART(pred_in_component, edg_source, edg_target, is_transition,
                          transition_factor):
    SEAGL_weights = np.ones((len(edg_source),), dtype='float32')
    pred_transition = pred_in_component[edg_source] != pred_in_component[edg_target]
    components_x, in_component_x = libply_c.connected_comp(pred_in_component.shape[0],
                                                           edg_source.astype('uint32'), edg_target.astype('uint32'),
                                                           (is_transition + pred_transition == 0).astype('uint8'), 0)

    edg_transition = is_transition.nonzero()[0]
    edg_source_trans = edg_source[edg_transition]
    edg_target_trans = edg_target[edg_transition]

    comp_x_weight = [len(c) for c in components_x]
    n_compx = len(components_x)

    edg_id = np.min((in_component_x[edg_source_trans], in_component_x[edg_target_trans]), 0) * n_compx + np.max(
        (in_component_x[edg_source_trans], in_component_x[edg_target_trans]), 0)

    edg_id_unique, in_edge_id, sedg_weight = np.unique(edg_id, return_index=True, return_counts=True)

    for i_edg in range(len(in_edge_id)):
        i_com_1 = in_component_x[edg_source_trans[in_edge_id[i_edg]]]
        i_com_2 = in_component_x[edg_target_trans[in_edge_id[i_edg]]]
        weight = min(comp_x_weight[i_com_1], comp_x_weight[i_com_2]) / sedg_weight[i_edg] * transition_factor
        corresponding_trans_edg = edg_transition[(
                (in_component_x[edg_source_trans] == i_com_1) * (in_component_x[edg_target_trans] == i_com_2) + (
                in_component_x[edg_target_trans] == i_com_1) * (in_component_x[edg_source_trans] == i_com_2))]
        SEAGL_weights[corresponding_trans_edg] = SEAGL_weights[corresponding_trans_edg] + weight
    return SEAGL_weights


def compute_partition(embeddings, edg_source, edg_target, diff, xyz):
    edge_weight_threshold = -0.5
    spatial_emb = 0.1
    reg_strength = 0.8
    k_nn_adj = 5
    CP_cutoff = 25
    edg_source = edg_source.astype('uint32')
    edg_target = edg_target.astype('uint32')
    edge_weight = torch.exp(diff * edge_weight_threshold).detach().cpu().numpy() / np.exp(
        edge_weight_threshold)

    ver_value = np.zeros((embeddings.shape[0], 0), dtype='f4')
    ver_value = np.hstack((ver_value, embeddings.detach().cpu().numpy()))
    ver_value = np.hstack((ver_value, spatial_emb * xyz))

    pred_components, pred_in_component = libcp.cutpursuit(ver_value,
                                                          edg_source, edg_target,
                                                          edge_weight,
                                                          reg_strength / (4 * k_nn_adj),
                                                          cutoff=CP_cutoff, spatial=True, weight_decay=0.7)
    return pred_components, pred_in_component


def compute_loss(diff, is_transition, weights_loss):
    intra_edg = is_transition == 0

    delta = 0.2
    loss1 = delta * (weights_loss[intra_edg] * (torch.sqrt(1 + diff[intra_edg] / delta ** 2) - 1)).sum()

    inter_edg = is_transition == 1

    loss2 = (zhang(torch.sqrt(diff[inter_edg] + 1e-10), weights_loss[inter_edg])).sum()

    return loss1, loss2


def compute_dist(embeddings, edg_source, edg_target, dist_type):
    """ Compute distance between embeddings.
    :param embeddings: embeddings
    :param edg_source: source edges
    :param edg_target: target edges
    :param dist_type: distance type
    """
    if dist_type == 'euclidian':
        dist = ((embeddings[edg_source, :] - embeddings[edg_target, :]) ** 2).sum(1)
    elif dist_type == 'intrinsic':
        smoothness = 0.999
        dist = (torch.acos((embeddings[edg_source, :] * embeddings[edg_target, :]).sum(1) * smoothness) - torch.arccos(
            smoothness)) / (torch.arccos(-smoothness) - torch.arccos(smoothness)) * 3.141592
    elif dist_type == 'scalar':
        dist = (embeddings[edg_source, :] * embeddings[edg_target, :]).sum(1) - 1
    else:
        raise ValueError(f'{dist_type} is an unknown distance type')
    return dist
