import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))
import libcp

from pointnet import PointNet
from graph import create_s3dis_datasets


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


def compute_partition(args, embeddings, edg_source, edg_target, diff, xyz=0):
    edge_weight = np.ones_like(edg_source).astype('f4')
    if args.edge_weight_threshold > 0:
        edge_weight[diff > 1] = args.edge_weight_threshold
    if args.edge_weight_threshold < 0:
        edge_weight = torch.exp(diff * args.edge_weight_threshold).detach().cpu().numpy() / np.exp(
            args.edge_weight_threshold)

    ver_value = np.zeros((embeddings.shape[0], 0), dtype='f4')
    use_spatial = 0
    ver_value = np.hstack((ver_value, embeddings.detach().cpu().numpy()))
    if args.spatial_emb > 0:
        ver_value = np.hstack((ver_value, args.spatial_emb * xyz))  # * math.sqrt(args.reg_strength)))
        use_spatial = 1  # !!!

    pred_components, pred_in_component = libcp.cutpursuit(ver_value,
                                                          edg_source.astype('uint32'), edg_target.astype('uint32'),
                                                          edge_weight,
                                                          args.reg_strength / (4 * args.k_nn_adj),
                                                          cutoff=args.CP_cutoff, spatial=use_spatial, weight_decay=0.7)
    return pred_components, pred_in_component


if __name__ == '__main__':
    model_file = '/home/ales/Thesis/ALVE-3D/models/pretrained/cv1/model.pth.tar'
    features_file = '/home/ales/Thesis/ALVE-3D/data/S3DIS/features_supervision/Area_1/conferenceRoom_1.h5'

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointNet(num_features=6, num_global_features=7, out_features=4)
    model.to(device)

    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # Load features

    # Compute distance

    # Compute partition

    # Visualize partition
