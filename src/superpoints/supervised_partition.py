import os
import sys
import ast
import math
import time
import random
import logging
import argparse

sys.path.append(os.path.join('/opt/conda/envs/ALVE-3D/lib/python3.10/site-packages/'))

import json
import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torchnet as tnt
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.superpoints.pointnet import PointNet
from src.superpoints.graph import compute_sp_graph, create_s3dis_datasets, graph_collate
from src.superpoints.provider import perfect_prediction, write_spg

from src.superpoints.metrics import compute_boundary_precision, compute_boundary_recall, ConfusionMatrix
from src.superpoints.losses import compute_weight_loss, compute_partition, relax_edge_binary, compute_loss, compute_dist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    # Dataset
    parser.add_argument('--dataset', default='s3dis', help='Dataset name: sema3d|s3dis|vkitti')
    parser.add_argument('--cvfold', default=1, type=int,
                        help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
    parser.add_argument('--db_train_name', default='trainval', help='Training set (Sema3D)')
    parser.add_argument('--db_test_name', default='testred', help='Test set (Sema3D)')
    parser.add_argument('--ROOT_PATH', default='data/S3DIS')
    parser.add_argument('--odir', default='models/pretrained', help='folder for saving the trained model')
    parser.add_argument('--spg_out', default=1, type=int,
                        help='wether to compute the SPG for linking with the SPG semantic segmentation method')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int,
                        help='Num subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=10, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int,
                        help='Average logits obtained over runs with different seeds')
    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float,
                        help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[20,35,45]',
                        help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float,
                        help='Element-wise clipping of gradient. If 0, does not clip')
    # Point cloud processing
    parser.add_argument('--pc_attribs', default='',
                        help='Point attributes fed to PointNets, if empty then all possible.')
    parser.add_argument('--pc_augm_scale', default=2, type=float,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')
    # Point net
    parser.add_argument('--ptn_embedding', default='ptn',
                        help='configuration of the learned cloud emebdder (ptn): uses PointNets '
                             'for vertices embeddings. no other options so far :)')
    parser.add_argument('--ptn_widths', default='[[32,128], [34,32,32,4]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[16,64],[32,16]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--use_color', default='rgb',
                        help='How to use color in the local cloud embedding : rgb, lab or no')
    parser.add_argument('--ptn_nfeat_stn', default=2, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_norm', default='batch',
                        help='Type of norm layers in PointNets, "batch or "layer" or "group"')
    parser.add_argument('--ptn_n_group', default=2, type=int,
                        help='Number of groups in groupnorm. Only compatible with ptn_norm=group')
    parser.add_argument('--stn_as_global', default=1, type=int,
                        help='Wether to use the STN output as a global variable')
    parser.add_argument('--global_feat', default='eXYrgb', help='Use rgb to embed points')
    parser.add_argument('--use_rgb', default=1, type=int,
                        help='Wether to use radiometry value to use for cloud embeding')
    parser.add_argument('--ptn_mem_monger', default=0, type=int,
                        help='Bool, save GPU memory by recomputing PointNets in back propagation.')

    # Loss
    parser.add_argument('--loss_weight', default='crosspartition',
                        help='[none, proportional, sqrt, seal, crosspartition] which loss weighting scheme to choose '
                             'to train the model. unweighted: use classic cross_entropy loss, proportional: '
                             'weight inversely by transition count,  SEAL: use SEAL loss as proposed in '
                             'https://jankautz.com/publications/LearningSuperpixels_CVPR2018.pdf, crosspartition : '
                             'our crosspartition weighting scheme')
    parser.add_argument('--loss', default='TVH_zhang',
                        help='Structure of the loss : first term for intra edge (chose from : tv, laplacian, '
                             'TVH (pseudo-huber)), second one for interedge (chose from: zhang, scad, tv)')
    parser.add_argument('--transition_factor', default=5, type=float,
                        help='Weight for transition edges in the graph structured contrastive loss')
    parser.add_argument('--dist_type', default='euclidian',
                        help='[euclidian, intrisic, scalar] How to measure the distance between embeddings')

    # Graph-Clustering
    parser.add_argument('--ver_value', default='ptn',
                        help='what value to use for vertices (ptn): uses PointNets, (geof) : '
                             'uses geometric features, (xyz) uses position, (rgb) uses color')
    parser.add_argument('--max_ver_train', default=1e3, type=int,
                        help='Size of the subgraph taken in each point cloud for the training')
    parser.add_argument('--k_nn_adj', default=5, type=int, help='number of neighbors for the adjacency graph')
    parser.add_argument('--k_nn_local', default=20, type=int, help='number of neighbors to describe the local geometry')
    parser.add_argument('--reg_strength', default=0.1, type=float,
                        help='Regularization strength or the generalized minimum partition problem.')
    parser.add_argument('--CP_cutoff', default=25, type=int,
                        help='Minimum accepted component size in cut pursuit. '
                             'if negative, chose with respect tot his number '
                             'and the reg_strength as explained in the paper')
    parser.add_argument('--spatial_emb', default=0.2, type=float,
                        help='Weight of xyz in the spatial embedding. When 0 : no xyz')
    parser.add_argument('--edge_weight_threshold', default=-0.5, type=float,
                        help='Edge weight value when diff>1. '
                             'if negative, then switch to weight = exp(-diff * edge_weight_threshold)')

    # Metrics
    parser.add_argument('--BR_tolerance', default=1, type=int,
                        help='How far an edge must be from an actual transition to be considered a true positive')

    args = parser.parse_args()

    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)
    args.learned_embeddings = ('ptn' in args.ver_value) or args.ver_value == 'xyz'
    if args.CP_cutoff < 0:  # adaptive cutoff: strong regularization will set a larger cutoff
        args.CP_cutoff = int(max(-args.CP_cutoff / 2, -args.CP_cutoff / 2 * np.log(args.reg_strength) - args.CP_cutoff))

    return args


class FolderHierarchy:
    SPG_FOLDER = "superpoint_graphs"
    EMBEDDINGS_FOLDER = "embeddings"
    SCALAR_FOLDER = "scalars"
    MODEL_FILE = "model.pth.tar"

    def __init__(self, output_dir, dataset_name, root_dir, cv_fold):
        self._root = root_dir
        if dataset_name == 's3dis':
            self._output_dir = os.path.join(output_dir, 'cv' + str(cv_fold))
            self._folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
        elif dataset_name == 'sema3d':
            self._output_dir = os.path.join(output_dir, 'best')
            self._folders = ["train/", "test_reduced/", "test_full/"]
        elif dataset_name == 'vkitti':
            self._output_dir = os.path.join(output_dir, 'cv' + str(cv_fold))
            self._folders = ["01/", "02/", "03/", "04/", "05/", "06/"]

        os.makedirs(self._output_dir, exist_ok=True)

        self._spg_folder = self._create_folder(self.SPG_FOLDER)
        self._emb_folder = self._create_folder(self.EMBEDDINGS_FOLDER)
        self._scalars = self._create_folder(self.SCALAR_FOLDER)

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def emb_folder(self):
        return self._emb_folder

    @property
    def spg_folder(self):
        return self._spg_folder

    @property
    def scalars(self):
        return self._scalars

    @property
    def model_path(self):
        return os.path.join(self._output_dir, self.MODEL_FILE)

    def _create_folder(self, property_name):
        folder = os.path.join(self._root, property_name)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        return folder


def main(args: argparse.Namespace):
    stats = []
    random.seed(0)
    with wandb.init(project='Evaluate partitioning'):
        root = os.path.join(args.ROOT_PATH)
        folder_hierarchy = FolderHierarchy(args.odir, args.dataset, root, args.cvfold)

        dbinfo = get_s3dis_info()

        # Create the datasets
        print('Creating datasets...')
        train_ds, val_ds = create_s3dis_datasets(args)
        print(f'Train dataset size: {len(train_ds)}')
        print(f'Val dataset size: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=graph_collate,
                                  num_workers=args.nworkers,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=graph_collate, num_workers=args.nworkers,
                                shuffle=False, drop_last=False)

        model = PointNet(num_features=6, num_global_features=7, out_features=4)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps,
                                                         gamma=args.lr_decay, last_epoch=args.start_epoch - 1)

        def train():
            model.train()

            loss_meter = tnt.meter.AverageValueMeter()
            n_clusters_meter = tnt.meter.AverageValueMeter()

            t0 = time.time()

            for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(
                    tqdm(train_loader)):

                # Move to device
                clouds, clouds_global, nei = clouds_data
                is_transition = is_transition.to(device)
                objects = objects.to(device)
                clouds = clouds.to(device)
                clouds_global = clouds_global.to(device)

                t_loader = 1000 * (time.time() - t0)
                optimizer.zero_grad()
                t0 = time.time()

                # Compute embeddings
                embeddings = model(clouds, clouds_global)

                # Compute loss
                diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)
                weights_loss, pred_comp, in_comp = compute_weight_loss(args, embeddings, objects, edg_source,
                                                                       edg_target,
                                                                       is_transition, diff, True, xyz)
                loss1, loss2 = compute_loss(args, diff, is_transition, weights_loss)

                wandb.log({"First Loss - train": loss1.item(), "Second Loss - train": loss2.item()})

                factor = 1000  # scaling for better usage of float precision

                loss = (loss1 + loss2) / weights_loss.shape[0] * factor

                loss.backward()

                if args.grad_clip > 0:
                    for p in model.parameters():
                        p.grad.data.clamp_(-args.grad_clip * factor, args.grad_clip * factor)

                optimizer.step()

                t_trainer = 1000 * (time.time() - t0)
                loss_meter.add(loss.item() / factor)  # /weights_loss.mean().item())
                n_clusters_meter.add(embeddings.shape[0] / len(pred_comp))

                logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.item() / factor, t_loader,
                              t_trainer)

                wandb.log({"Batch loss - train": loss.item() / factor, "Loader time - train": t_loader,
                           "Trainer time - train": t_trainer})

                t0 = time.time()

            wandb.log({"Epoch loss - train": loss_meter.value()[0],
                       "Epoch n_clusters - train": n_clusters_meter.value()[0]})

            return loss_meter.value()[0], n_clusters_meter.value()[0]

        def evaluate():
            """ Evaluated model on test set """
            model.eval()

            with torch.no_grad():

                loss_meter = tnt.meter.AverageValueMeter()
                n_clusters_meter = tnt.meter.AverageValueMeter()
                BR_meter = tnt.meter.AverageValueMeter()
                BP_meter = tnt.meter.AverageValueMeter()
                CM_classes = ConfusionMatrix(dbinfo['classes'])

                # iterate over dataset in batches
                for bidx, (
                        fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(
                    tqdm(val_loader)):

                    # Move to device
                    clouds, clouds_global, nei = clouds_data
                    is_transition = is_transition.to(device)
                    objects = objects.to(device)
                    clouds = clouds.to(device)
                    clouds_global = clouds_global.to(device)

                    embeddings = model(clouds, clouds_global)

                    diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)

                    if len(is_transition) > 1:
                        weights_loss, pred_components, pred_in_component = compute_weight_loss(args, embeddings,
                                                                                               objects,
                                                                                               edg_source, edg_target,
                                                                                               is_transition, diff,
                                                                                               True,
                                                                                               xyz)
                        loss1, loss2 = compute_loss(args, diff, is_transition, weights_loss)
                        loss = (loss1 + loss2) / weights_loss.shape[0]
                        pred_transition = pred_in_component[edg_source] != pred_in_component[edg_target]
                        per_pred = perfect_prediction(pred_components, labels)
                        CM_classes.count_predicted_batch(labels[:, 1:], per_pred)
                    else:
                        loss = 0

                    if len(is_transition) > 1:
                        loss_meter.add(loss.item())  # /weights_loss.sum().item())
                        is_transition = is_transition.cpu().numpy()
                        n_clusters_meter.add(len(pred_components))
                        BR_meter.add((is_transition.sum()) * compute_boundary_recall(is_transition,
                                                                                     relax_edge_binary(pred_transition,
                                                                                                       edg_source,
                                                                                                       edg_target,
                                                                                                       xyz.shape[0],
                                                                                                       args.BR_tolerance)),
                                     n=is_transition.sum())
                        BP_meter.add((pred_transition.sum()) * compute_boundary_precision(
                            relax_edge_binary(is_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance),
                            pred_transition), n=pred_transition.sum())
            CM = CM_classes.confusion_matrix
            return loss_meter.value()[0], n_clusters_meter.value()[0], 100 * CM.trace() / CM.sum(), BR_meter.value()[0], \
                BP_meter.value()[0]

        def evaluate_final():
            """ Evaluated model on test set """

            print("Final evaluation")
            model.eval()

            loss_meter = tnt.meter.AverageValueMeter()
            n_clusters_meter = tnt.meter.AverageValueMeter()
            confusion_matrix_classes = ConfusionMatrix(dbinfo['classes'])
            confusion_matrix_BR = ConfusionMatrix(2)
            confusion_matrix_BP = ConfusionMatrix(2)

            with torch.no_grad():

                # iterate over dataset in batches
                for bidx, (
                        fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(
                    tqdm(val_loader)):

                    # Move to device
                    clouds, clouds_global, nei = clouds_data
                    is_transition = is_transition.to(device, non_blocking=True)
                    objects = objects.to(device, non_blocking=True)
                    clouds = clouds.to(device, non_blocking=True)
                    clouds_global = clouds_global.to(device, non_blocking=True)

                    embeddings = model(clouds, clouds_global)

                    diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)

                    pred_components, pred_in_component = compute_partition(args, embeddings, edg_source, edg_target,
                                                                           diff,
                                                                           xyz)

                    if len(is_transition) > 1:
                        pred_transition = pred_in_component[edg_source] != pred_in_component[edg_target]
                        is_transition = is_transition.cpu().numpy()

                        n_clusters_meter.add(len(pred_components))

                        per_pred = perfect_prediction(pred_components, labels)
                        confusion_matrix_classes.count_predicted_batch(labels[:, 1:], per_pred)
                        confusion_matrix_BR.count_predicted_batch_hard(is_transition,
                                                                       relax_edge_binary(pred_transition, edg_source,
                                                                                         edg_target, xyz.shape[0],
                                                                                         args.BR_tolerance).astype(
                                                                           'uint8'))
                        confusion_matrix_BP.count_predicted_batch_hard(
                            relax_edge_binary(is_transition, edg_source, edg_target, xyz.shape[0], args.BR_tolerance),
                            pred_transition.astype('uint8'))

                    if args.spg_out:
                        graph_sp = compute_sp_graph(xyz, 100, pred_in_component, pred_components, labels,
                                                    dbinfo["classes"])
                        spg_file = os.path.join(folder_hierarchy.spg_folder, fname[0])
                        if not os.path.exists(os.path.dirname(spg_file)):
                            os.makedirs(os.path.dirname(spg_file))
                        try:
                            os.remove(spg_file)
                        except OSError:
                            pass
                        write_spg(spg_file, graph_sp, pred_components, pred_in_component)

                        # Debugging purpose - write the embedding file and an exemple of scalar files
                        # if bidx % 0 == 0:
                        #     embedding2ply(os.path.join(folder_hierarchy.emb_folder , fname[0][:-3] + '_emb.ply'), xyz, embeddings.detach().cpu().numpy())
                        #     scalar2ply(os.path.join(folder_hierarchy.scalars , fname[0][:-3] + '_elevation.ply') , xyz, clouds_data[1][:,1].cpu())
                        #     edg_class = is_transition + 2*pred_transition
                        #     edge_class2ply2(os.path.join(folder_hierarchy.emb_folder , fname[0][:-3] + '_transition.ply'), edg_class, xyz, edg_source, edg_target)

                if len(is_transition) > 1:
                    res_name = folder_hierarchy.output_dir + '/res.h5'
                    res_file = h5py.File(res_name, 'w')
                    res_file.create_dataset('confusion_matrix_classes'
                                            , data=confusion_matrix_classes.confusion_matrix, dtype='uint64')
                    res_file.create_dataset('confusion_matrix_BR'
                                            , data=confusion_matrix_BR.confusion_matrix, dtype='uint64')
                    res_file.create_dataset('confusion_matrix_BP'
                                            , data=confusion_matrix_BP.confusion_matrix, dtype='uint64')
                    res_file.create_dataset('n_clusters'
                                            , data=n_clusters_meter.value()[0], dtype='uint64')
                    res_file.close()

            return

        # for epoch in range(args.start_epoch, args.epochs):
        #     if not args.learned_embeddings:
        #         break
        #     print('Epoch {}/{} ({}):'.format(epoch, args.epochs, folder_hierarchy.output_dir))
        #
        #     loss, n_sp = train()
        #     scheduler.step()
        #
        #     if (epoch + 1) % args.test_nth_epoch == 0:  # or epoch+1==args.epochs:
        #         loss_test, n_clusters_test, ASA_test, BR_test, BP_test = evaluate()
        #         print(
        #             '-> Train loss: %1.5f - Test Loss: %1.5f  |  n_clusters:  %5.1f  |  ASA: %3.2f %%  |  Test BR: %3.2f %%  |  BP : %3.2f%%' % (
        #                 loss, loss_test, n_clusters_test, ASA_test, BR_test, BP_test))
        #     else:
        #         loss_test, n_clusters_test, ASA_test, BR_test, BP_test = 0, 0, 0, 0, 0
        #         print('-> Train loss: %1.5f  superpoints size : %5.0f' % (loss, n_sp))
        #
        #     stats.append({'epoch': epoch, 'loss': loss, 'loss_test': loss_test, 'n_clusters_test': n_clusters_test,
        #                   'ASA_test': ASA_test, 'BR_test': BR_test, 'BP_test': BP_test})
        #
        #     with open(os.path.join(folder_hierarchy.output_dir, 'trainlog.json'), 'w') as outfile:
        #         json.dump(stats, outfile, indent=4)
        #
        #     if epoch % args.save_nth_epoch == 0 or epoch == args.epochs - 1:
        #         model_name = 'model.pth.tar'
        #         print("Saving model to " + model_name)
        #         model_name = 'model.pth.tar'
        #         torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(),
        #                     'optimizer': optimizer.state_dict()},
        #                    os.path.join(folder_hierarchy.output_dir, model_name))
        #
        #     if math.isnan(loss): break
        #
        # evaluate_final()

        # ==========================================================================================

        for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(
                tqdm(val_loader)):
            clouds, clouds_global, nei = clouds_data
            is_transition = is_transition.to(device, non_blocking=True)
            objects = objects.to(device, non_blocking=True)
            clouds = clouds.to(device, non_blocking=True)
            clouds_global = clouds_global.to(device, non_blocking=True)

            embeddings = model(clouds, clouds_global)

            diff = compute_dist(embeddings, edg_source, edg_target, args.dist_type)

            pred_components, pred_in_component = compute_partition(args, embeddings, edg_source, edg_target, diff, xyz)

            # Map colors to components
            print(max(pred_in_component))
            color_map = instances_color_map()
            pred_components_color = color_map[pred_in_component]

            cloud = np.concatenate([xyz, pred_components_color * 255], axis=1)

            # Log statistics
            wandb.log({'Point Cloud': wandb.Object3D(cloud)})

            print(f'\nLogged scan: {fname[0]}')

            # graph_sp = compute_sp_graph(xyz, 100, pred_in_component, pred_components, labels, 13)
            #
            # spg_file = os.path.join(folder_hierarchy.spg_folder, fname[0])
            #
            # write_spg(spg_file, graph_sp, pred_components, pred_in_component)


def instances_color_map():
    # make instance colors
    max_inst_id = 100000
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    color_map[0] = np.full(3, 0.1)
    return color_map


def get_s3dis_info():
    # for now, no edge attributes
    return {
        'classes': 13,
        'inv_class_map': {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'column', 4: 'beam', 5: 'window', 6: 'door',
                          7: 'table', 8: 'chair', 9: 'bookcase', 10: 'sofa', 11: 'board', 12: 'clutter'},
    }


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
