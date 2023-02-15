import os
import argparse

import h5py
import torch
import functools
import numpy as np
import torchnet as tnt
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor

from src.superpoints.ply_c import libply_c
from src.superpoints.provider import read_s3dis_format


def main():
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    parser.add_argument('--ROOT_PATH', default='/home/ales/Datasets/S3DIS')

    parser.add_argument('--compute_geof', default=1, type=int,
                        help='compute hand-crafted features of the local geometry')
    parser.add_argument('--k_nn_local', default=20, type=int, help='number of neighbors to describe the local geometry')
    parser.add_argument('--k_nn_adj', default=5, type=int, help='number of neighbors for the adjacency graph')
    parser.add_argument('--voxel_width', default=0.00, type=float, help='voxel size when subsampling (in m)')
    parser.add_argument('--plane_model', default=True, type=bool, help='uses a simple plane model to derive elevation')
    parser.add_argument('--ver_batch', default=5000000, type=int, help='batch size for reading large files')
    args = parser.parse_args()

    folders = ['Area_2']
    n_labels = 13

    for folder in folders:
        print(f'=================\n{folder}\n================')

        data_folder = os.path.join(args.ROOT_PATH, 'data', folder)
        str_folder = os.path.join(args.ROOT_PATH, 'features_supervision', folder)

        os.makedirs(str_folder, exist_ok=True)

        # List all directories in the data folder
        samples = [os.path.join(data_folder, o) for o in os.listdir(data_folder) if
                   os.path.isdir(os.path.join(data_folder, o))]

        for i, sample in enumerate(samples):
            name = os.path.splitext(os.path.basename(sample))[0]
            print(f'[{i + 1}/{len(samples)}] {name}...')

            # Read the point cloud and corresponding labels
            data_file = os.path.join(data_folder, name, name + ".txt")
            str_file = os.path.join(str_folder, name + '.h5')

            if os.path.isfile(str_file):
                print("\tGraph structure already computed - delete for update...")
            else:
                print("Computing graph structure...")

                # Read sample
                xyz, rgb, labels, objects = read_s3dis_format(data_file)

                # Prune the point cloud
                if args.voxel_width > 0:
                    n_objects = int(np.max(objects) + 1)
                    xyz, rgb, labels, objects = libply_c.prune(xyz, args.voxel_width, rgb, labels,
                                                               objects, n_labels, n_objects)
                n_ver = xyz.shape[0]

                print('\t- computing NN structure')
                graph_nn, local_neighbors = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_local)
                is_transition = objects[graph_nn['source']] != objects[graph_nn['target']]

                print('\t- computing local geometry')
                geof = libply_c.compute_geof(xyz, local_neighbors, args.k_nn_local).astype(np.float32)
                geof[:, 3] = 2 * geof[:, 3]

                print('\t- computing elevation')
                if args.plane_model:
                    low_points = (xyz[:, 2] - xyz[:, 2].min() < 0.5).nonzero()[0]
                    reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
                    elevation = xyz[:, 2] - reg.predict(xyz[:, :2])

                else:
                    elevation = xyz[:, 2] - xyz[:, 2].min()

                print('\t -computing normalized coordinates')
                ma, mi = np.max(xyz[:, :2], axis=0, keepdims=True), np.min(xyz[:, :2], axis=0, keepdims=True)
                xyn = (xyz[:, :2] - mi) / (ma - mi)

                print(f'\t- writing structure to {str_file}')
                write_structure(str_file, xyz, rgb, graph_nn, local_neighbors.reshape([n_ver, args.k_nn_local]),
                                is_transition, labels, objects, geof, elevation, xyn)


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


def compute_graph_nn_2(xyz, k_nn1, k_nn2):
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


def compute_sp_graph(xyz, d_max, in_component, components, labels, n_labels):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    # ---compute delaunay triangulation---
    tri = Delaunay(xyz)
    # interface select the edges between different components
    # edgx and edgxr converts from tetrahedrons to edges
    # done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r, edg5r, edg6r))
    del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)

    if d_max > 0:
        dist = np.sqrt(((xyz[edges[0, :]] - xyz[edges[1, :]]) ** 2).sum(1))
        edges = edges[:, dist < d_max]

    # ---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    # marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    # ---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='uint64')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_norm"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_delta_centroid"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_length_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_surface_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_volume_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_point_count_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                                                        , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp, :])
        graph["sp_point_count"][i_com] = len(comp)
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            graph["sp_length"][i_com] = 0
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        else:
            ev = np.linalg.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0])  # descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            try:
                graph["sp_length"][i_com] = ev[0]
            except TypeError:
                graph["sp_length"][i_com] = 0
            try:
                graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
            except TypeError:
                graph["sp_surface"][i_com] = 0
            try:
                graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            except TypeError:
                graph["sp_volume"][i_com] = 0
    # ---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        graph["source"][i_sedg] = com_source
        graph["target"][i_sedg] = com_target
        # ---compute the ratio features---
        graph["se_delta_centroid"][i_sedg, :] = graph["sp_centroids"][com_source, :] - graph["sp_centroids"][com_target,
                                                                                       :]
        graph["se_length_ratio"][i_sedg] = graph["sp_length"][com_source] / (graph["sp_length"][com_target] + 1e-6)
        graph["se_surface_ratio"][i_sedg] = graph["sp_surface"][com_source] / (graph["sp_surface"][com_target] + 1e-6)
        graph["se_volume_ratio"][i_sedg] = graph["sp_volume"][com_source] / (graph["sp_volume"][com_target] + 1e-6)
        graph["se_point_count_ratio"][i_sedg] = graph["sp_point_count"][com_source] / (
                graph["sp_point_count"][com_target] + 1e-6)
        # ---compute the offset set---
        delta = xyz_source - xyz_target
        if len(delta) > 1:
            graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
            graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            graph["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            graph["se_delta_mean"][i_sedg, :] = delta
            graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
            graph["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))
    return graph


def create_s3dis_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """
    # Load formatted clouds
    testlist, trainlist = [], []
    for n in range(1, 3):
        if n != args.cvfold:
            path = '{}/features_supervision/Area_{:d}/'.format(args.ROOT_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5"):
                    trainlist.append(path + fname)
    path = '{}/features_supervision/Area_{:d}/'.format(args.ROOT_PATH, args.cvfold)
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            testlist.append(path + fname)

    return tnt.dataset.ListDataset(trainlist,
                                   functools.partial(graph_loader, train=True, args=args, db_path=args.ROOT_PATH)), \
        tnt.dataset.ListDataset(testlist,
                                functools.partial(graph_loader, train=False, args=args, db_path=args.ROOT_PATH))


def subgraph_sampling(n_ver, edg_source, edg_target, max_ver):
    """ Select a subgraph of the input graph of max_ver verices"""
    return libply_c.random_subgraph(n_ver)


def graph_loader(entry, train, args, db_path, test_seed_offset=0, full_cpu=False):
    """ Load the point cloud and the graph structure """
    xyz, rgb, edg_source, edg_target, is_transition, local_geometry \
        , labels, objects, elevation, xyn = read_structure(entry, 'geof' in args.ver_value)
    short_name = entry.split(os.sep)[-2] + '/' + entry.split(os.sep)[-1]

    rgb = rgb / 255

    n_ver = np.shape(xyz)[0]
    n_edg = np.shape(edg_source)[0]

    selected_ver = np.full((n_ver,), True, dtype='?')
    selected_edg = np.full((n_edg,), True, dtype='?')

    # if train:
    #     xyz, rgb = augment_cloud_whole(args, xyz, rgb)

    subsample = False
    new_ver_index = []

    if train and (0 < args.max_ver_train < n_ver):
        subsample = True

        selected_edg, selected_ver = libply_c.random_subgraph(n_ver, edg_source.astype('uint32'),
                                                              edg_target.astype('uint32'), int(args.max_ver_train))
        # Change the type to bool
        selected_edg = selected_edg.astype(bool)
        selected_ver = selected_ver.astype(bool)

        new_ver_index = -np.ones((n_ver,), dtype=int)
        new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

        edg_source = new_ver_index[edg_source[selected_edg.astype('?')]]
        edg_target = new_ver_index[edg_target[selected_edg.astype('?')]]

        is_transition = is_transition[selected_edg]
        labels = labels[selected_ver,]
        objects = objects[selected_ver,]
        elevation = elevation[selected_ver]
        xyn = xyn[selected_ver,]

    if args.learned_embeddings:
        # we use point nets to embed the point clouds
        nei = local_geometry[selected_ver, :args.k_nn_local].astype('int64')

        clouds, clouds_global = [], []  # clouds_global is cloud global features. here, just the diameter + elevation

        clouds = xyz[nei,]
        # diameters = np.max(np.max(clouds,axis=1) - np.min(clouds,axis=1), axis = 1)
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - xyz[selected_ver, np.newaxis, :]) / (diameters[:, np.newaxis, np.newaxis] + 1e-10)

        if args.use_rgb:
            clouds = np.concatenate([clouds, rgb[nei,]], axis=2)

        clouds = clouds.transpose([0, 2, 1])

        clouds_global = diameters[:, None]
        if 'e' in args.global_feat:
            clouds_global = np.hstack((clouds_global, elevation[:, None]))
        if 'rgb' in args.global_feat:
            clouds_global = np.hstack((clouds_global, rgb[selected_ver,]))
        if 'XY' in args.global_feat:
            clouds_global = np.hstack((clouds_global, xyn))
        if 'xy' in args.global_feat:
            clouds_global = np.hstack((clouds_global, xyz[selected_ver, :2]))
        # clouds_global = np.hstack((diameters[:,None], ((xyz[selected_ver,2] - min_z) / (max_z- min_z)-0.5)[:,None],np.zeros_like(rgb[selected_ver,])))

        # clouds_global = np.vstack((diameters, xyz[selected_ver,2])).T
    elif args.ver_value == 'geofrgb':
        # the embeddings are already computed
        clouds = np.concatenate([local_geometry, rgb[selected_ver,]], axis=1)
        clouds_global = np.array([0])
        nei = np.array([0])
    elif args.ver_value == 'geof':
        # the embeddings are already computed
        clouds = local_geometry
        clouds_global = np.array([0])
        nei = np.array([0])

    n_edg_selected = selected_edg.sum()

    nei = np.array([0])

    xyz = xyz[selected_ver,]
    is_transition = torch.from_numpy(is_transition)
    # labels = torch.from_numpy(labels)
    objects = torch.from_numpy(objects.astype('int64'))
    clouds = torch.from_numpy(clouds)
    clouds_global = torch.from_numpy(clouds_global)
    return short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, nei, xyz


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def graph_collate(batch):
    """ Collates a list of dataset samples into a single batch
    """
    short_name, edg_source, edg_target, is_transition, labels, objects, clouds, clouds_global, nei, xyz = list(
        zip(*batch))

    n_batch = len(short_name)
    batch_ver_size_cumsum = np.array([c.shape[0] for c in labels]).cumsum()
    batch_n_edg_cumsum = np.array([c.shape[0] for c in edg_source]).cumsum()
    batch_n_objects_cumsum = np.array([c.max() for c in objects]).cumsum()

    clouds = torch.cat(clouds, 0)
    clouds_global = torch.cat(clouds_global, 0)
    xyz = np.vstack(xyz)
    # if len(is_transition[0])>1:
    is_transition = torch.cat(is_transition, 0)
    labels = np.vstack(labels)

    edg_source = np.hstack(edg_source)
    edg_target = np.hstack(edg_target)
    nei = np.vstack(nei)
    # if len(is_transition[0]>1:
    objects = torch.cat(objects, 0)

    for i_batch in range(1, n_batch):
        edg_source[batch_n_edg_cumsum[i_batch - 1]:batch_n_edg_cumsum[i_batch]] += int(
            batch_ver_size_cumsum[i_batch - 1])
        edg_target[batch_n_edg_cumsum[i_batch - 1]:batch_n_edg_cumsum[i_batch]] += int(
            batch_ver_size_cumsum[i_batch - 1])
        # if len(objects)>1:
        objects[batch_ver_size_cumsum[i_batch - 1]:batch_ver_size_cumsum[i_batch], ] += int(
            batch_n_objects_cumsum[i_batch - 1])
        non_valid = (nei[batch_ver_size_cumsum[i_batch - 1]:batch_ver_size_cumsum[i_batch], ] == -1).nonzero()
        nei[batch_ver_size_cumsum[i_batch - 1]:batch_ver_size_cumsum[i_batch], ] += int(
            batch_ver_size_cumsum[i_batch - 1])
        nei[batch_ver_size_cumsum[i_batch - 1] + non_valid[0], non_valid[1]] = -1

    return short_name, edg_source, edg_target, is_transition, labels, objects, (clouds, clouds_global, nei), xyz


# ------------------------------------------------------------------------------
def show(clouds, k):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(clouds[k, :, 0], clouds[k, :, 1], clouds[k, :, 2])
    plt.show()


# ------------------------------------------------------------------------------
def read_embeddings(file_name):
    """
    read the input point cloud in a format ready for embedding
    """
    data_file = h5py.File(file_name, 'r')
    if 'embeddings' in data_file:
        embeddings = np.array(data_file['embeddings'], dtype='float32')
    else:
        embeddings = []
    if 'edge_weight' in data_file:
        edge_weight = np.array(data_file['edge_weight'], dtype='float32')
    else:
        edge_weight = []
    return embeddings, edge_weight


# ------------------------------------------------------------------------------
def write_embeddings(file_name, args, embeddings, edge_weight=[]):
    """
    save the embeddings and the edge weights
    """
    folder = args.ROOT_PATH + '/embeddings' + args.suffix + '/' + file_name.split('/')[0]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file_path = args.ROOT_PATH + '/embeddings' + args.suffix + '/' + file_name
    if os.path.isfile(file_path):
        data_file = h5py.File(file_path, 'r+')
    else:
        data_file = h5py.File(file_path, 'w')
    if len(embeddings) > 0 and not 'embeddings' in data_file:
        data_file.create_dataset('embeddings'
                                 , data=embeddings, dtype='float32')
    elif len(embeddings) > 0:
        data_file['embeddings'][...] = embeddings

    if len(edge_weight) > 0 and not 'edge_weight' in data_file:
        data_file.create_dataset('edge_weight', data=edge_weight, dtype='float32')
    elif len(edge_weight) > 0:
        data_file['edge_weight'][...] = edge_weight
    data_file.close()


if __name__ == '__main__':
    main()
