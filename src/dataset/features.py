import os
import sys

import h5py
import wandb
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from numpy.lib.recfunctions import structured_to_unstructured

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

from .utils import open_sequence
from .dataset import SemanticDataset
from src.laserscan import LaserScan
from src.ply_c import libply_c
from src.model.pointnet import PointNet
from src.kitti360.ply import read_ply
import libcp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_superpoints_2(cfg: DictConfig):
    # Load the dataset
    ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, sequences=[3])
    laser_scan = LaserScan(label_map=cfg.ds.learning_map, colorize=True, color_map=cfg.ds.color_map_train)

    # Load the model
    model = PointNet(num_features=6, num_global_features=7, out_features=4)
    model.to(device)

    checkpoint = torch.load(os.path.join(cfg.path.models, 'pretrained', 'cv1', 'model.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    print(f'Visualizing superpoints for sequence 3 ({len(ds)})')
    with torch.no_grad():
        with wandb.init(project='superpoint'):
            for i in tqdm(range(20, len(ds))):
                # Load data
                laser_scan.open_scan(ds.scans[i])
                laser_scan.open_label(ds.labels[i])

                xyz = laser_scan.points
                rgb = laser_scan.color
                labels = laser_scan.sem_label

                # Prune the data
                xyz, rgb, labels, o = libply_c.prune(xyz.astype('float32'), 0.1, rgb.astype('uint8'),
                                                     labels.astype('uint8'),
                                                     np.zeros(1, dtype='uint8'), 20, 0)

                # # Compute the nearest neighbors
                graph_nn, local_neighbors = compute_graph_nn(xyz, 5, 20)

                # Compute the elevation
                low_points = (xyz[:, 2] - xyz[:, 2].min() < 0.5).nonzero()[0]
                reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
                elevation = xyz[:, 2] - reg.predict(xyz[:, :2])

                # Compute the xyn (normalized x and y)
                ma, mi = np.max(xyz[:, :2], axis=0, keepdims=True), np.min(xyz[:, :2], axis=0, keepdims=True)
                xyn = (xyz[:, :2] - mi) / (ma - mi)

                nei = local_neighbors.reshape([xyz.shape[0], 20]).astype('int64')

                # Compute the local geometry
                clouds = xyz[nei,]
                diameters = np.sqrt(clouds.var(1).sum(1))
                clouds = (clouds - xyz[:, np.newaxis, :]) / (diameters[:, np.newaxis, np.newaxis] + 1e-10)
                clouds = np.concatenate([clouds, rgb[nei,]], axis=2)
                clouds = clouds.transpose([0, 2, 1])

                # Compute the global geometry
                clouds_global = np.hstack([diameters[:, np.newaxis], elevation[:, np.newaxis], rgb, xyn])

                # is_transition = torch.from_numpy(is_transition)
                # objects = torch.from_numpy(objects.astype('int64'))
                clouds = torch.from_numpy(clouds)
                clouds_global = torch.from_numpy(clouds_global)

                clouds = clouds.to(device, non_blocking=True)
                clouds_global = clouds_global.to(device, non_blocking=True)

                # embeddings = model(clouds, clouds_global)
                embeddings = model(clouds, clouds_global)
                source = graph_nn['source'].astype('int64')
                target = graph_nn['target'].astype('int64')
                diff = ((embeddings[source, :] - embeddings[target, :]) ** 2).sum(1)

                pred_components, pred_in_component = compute_partition(embeddings, graph_nn['source'],
                                                                       graph_nn['target'], diff,
                                                                       xyz)

                color_map = instances_color_map()
                pred_components_color = color_map[pred_in_component]

                cloud = np.concatenate([xyz, pred_components_color * 255], axis=1)

                # Log statistics
                wandb.log({'Point Cloud': wandb.Object3D(cloud)})


def visualize_superpoints(cfg: DictConfig):
    window_file = '/home/kuceral4/ALVE-3D/data/KITTI-360/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000599_0000000846.ply'
    static_window = read_ply(window_file)

    static_points = structured_to_unstructured(static_window[['x', 'y', 'z']])
    static_colors = structured_to_unstructured(static_window[['red', 'green', 'blue']]) / 255

    semantic = structured_to_unstructured(static_window[['semantic']])

    # Load the model
    model = PointNet(num_features=6, num_global_features=7, out_features=4, memory_size=10000)
    model.to(device)

    checkpoint = torch.load(os.path.join(cfg.path.models, 'pretrained', 'cv1', 'model.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    with wandb.init(project='superpoint'):
        xyz = static_points
        rgb = static_colors

        # Prune the data
        # xyz, rgb, labels, o = libply_c.prune(xyz.astype('float32'), 0.15, rgb.astype('uint8'),
        #                                      np.ones(xyz.shape[0], dtype='uint8'),
        #                                      np.zeros(1, dtype='uint8'), 20, 0)

        # # Compute the nearest neighbors
        graph_nn, local_neighbors = compute_graph_nn(xyz, 5, 20)

        # Compute the elevation
        low_points = (xyz[:, 2] - xyz[:, 2].min() < 0.5).nonzero()[0]
        reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
        elevation = xyz[:, 2] - reg.predict(xyz[:, :2])

        # Compute the xyn (normalized x and y)
        ma, mi = np.max(xyz[:, :2], axis=0, keepdims=True), np.min(xyz[:, :2], axis=0, keepdims=True)
        xyn = (xyz[:, :2] - mi) / (ma - mi)

        nei = local_neighbors.reshape([xyz.shape[0], 20]).astype('int64')

        # Compute the local geometry
        clouds = xyz[nei,]
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - xyz[:, np.newaxis, :]) / (diameters[:, np.newaxis, np.newaxis] + 1e-10)
        clouds = np.concatenate([clouds, rgb[nei,]], axis=2)
        clouds = clouds.transpose([0, 2, 1])

        # Compute the global geometry
        clouds_global = np.hstack([diameters[:, np.newaxis], elevation[:, np.newaxis], rgb, xyn])

        # is_transition = torch.from_numpy(is_transition)
        # objects = torch.from_numpy(objects.astype('int64'))
        clouds = torch.from_numpy(clouds)
        clouds_global = torch.from_numpy(clouds_global)

        clouds = clouds.to(device, non_blocking=True)
        clouds_global = clouds_global.to(device, non_blocking=True)

        # embeddings = cloud_embedder.run_batch(model, clouds, clouds_global)
        with torch.no_grad():
            embeddings = model(clouds, clouds_global)

        source = graph_nn['source'].astype('int64')
        target = graph_nn['target'].astype('int64')
        diff = ((embeddings[source, :] - embeddings[target, :]) ** 2).sum(1)

        pred_components, pred_in_component = compute_partition(embeddings, graph_nn['source'],
                                                               graph_nn['target'], diff,
                                                               xyz)

        color_map = instances_color_map()
        pred_components_color = color_map[pred_in_component]

        cloud = np.concatenate([xyz, pred_components_color * 255], axis=1)

        # Log statistics
        wandb.log({'Point Cloud': wandb.Object3D(cloud)})


def instances_color_map():
    # make instance colors
    max_inst_id = 100000
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    color_map[0] = np.full(3, 0.1)
    return color_map


def compute_partition(embeddings, edg_source, edg_target, diff, xyz=0):
    edge_weight_threshold = -0.5
    spatial_emb = 0.2
    reg_strength = 0.1
    k_nn_adj = 5
    CP_cutoff = 25
    edge_weight = torch.exp(diff * edge_weight_threshold).detach().cpu().numpy() / np.exp(
        edge_weight_threshold)

    ver_value = np.zeros((embeddings.shape[0], 0), dtype='f4')
    ver_value = np.hstack((ver_value, embeddings.detach().cpu().numpy()))
    ver_value = np.hstack((ver_value, spatial_emb * xyz))

    pred_components, pred_in_component = libcp.cutpursuit(ver_value,
                                                          edg_source.astype('uint32'), edg_target.astype('uint32'),
                                                          edge_weight,
                                                          reg_strength / (4 * k_nn_adj),
                                                          cutoff=CP_cutoff, spatial=True, weight_decay=0.7)
    return pred_components, pred_in_component


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


def compute_graph_nn(xyz, k_nn1, k_nn2):
    """compute simultaneously 2 knn structures,
    only saves target for knn2
    """
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])

    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = np.array(nn.kneighbors(xyz))[..., 1:]

    # ---knn2---
    target2 = (neighbors.flatten()).astype('uint32')

    # ---knn1----
    neighbors = neighbors[:, :k_nn1]
    distances = distances[:, :k_nn1]

    indices = np.arange(0, n_ver)[..., np.newaxis]
    source = np.zeros((n_ver, k_nn1)) + indices

    graph['source'] = source.flatten().astype(np.uint32)
    graph["target"] = neighbors.flatten().astype(np.uint32)
    graph["distances"] = distances.flatten().astype(np.float32)
    return graph, target2


def write_structure(file_name, xyz, rgb, graph_nn, target_local_geometry, is_transition, labels, objects, geof,
                    elevation, xyn):
    """
    save the input point cloud in a format ready for embedding
    """
    # store transition and non-transition edges in two different contiguous memory blocks

    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    data_file.create_dataset('rgb', data=rgb, dtype='float32')
    data_file.create_dataset('elevation', data=elevation, dtype='float32')
    data_file.create_dataset('xyn', data=xyn, dtype='float32')
    data_file.create_dataset('source', data=graph_nn["source"], dtype='int')
    data_file.create_dataset('target', data=graph_nn["target"], dtype='int')
    data_file.create_dataset('is_transition', data=is_transition, dtype='uint8')
    data_file.create_dataset('target_local_geometry', data=target_local_geometry, dtype='uint32')
    data_file.create_dataset('objects', data=objects, dtype='uint32')
    if len(geof) > 0:
        data_file.create_dataset('geof', data=geof, dtype='float32')
    if len(labels) > 0 and len(labels.shape) > 1 and labels.shape[1] > 1:
        data_file.create_dataset('labels', data=labels, dtype='int32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')


def read_structure(file_name, read_geof):
    """
    read the input point cloud in a format ready for embedding
    """
    data_file = h5py.File(file_name, 'r')
    xyz = np.array(data_file['xyz'], dtype='float32')
    rgb = np.array(data_file['rgb'], dtype='float32')
    elevation = np.array(data_file['elevation'], dtype='float32')
    xyn = np.array(data_file['xyn'], dtype='float32')
    edg_source = np.array(data_file['source'], dtype='int').squeeze()
    edg_target = np.array(data_file['target'], dtype='int').squeeze()
    is_transition = np.array(data_file['is_transition'])
    objects = np.array(data_file['objects'][()])
    labels = np.array(data_file['labels']).squeeze()
    if len(labels.shape) == 0:  # dirty fix
        labels = np.array([0])
    if len(is_transition.shape) == 0:  # dirty fix
        is_transition = np.array([0])
    if read_geof:  # geometry = geometric features
        local_geometry = np.array(data_file['geof'], dtype='float32')
    else:  # geometry = neighborhood structure
        local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')

    return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn
