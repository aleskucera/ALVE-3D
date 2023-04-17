# import os
#
# from omegaconf import DictConfig
#
# from .utils import open_sequence
#
#
# class SemanticKITTIConverter:
#     def __init__(self, cfg: DictConfig):
#
#         self.cfg = cfg
#         self.sequence = cfg.sequence if 'sequence' in cfg else 3
#
#         self.sequence_path = os.path.join(cfg.ds.path, 'sequences', f"{self.sequence:02d}")
#         self.scans, self.labels, self.poses = open_sequence(self.sequence_path)
#
#         self.window_ranges = create_window_ranges(self.scans)
#         splits = self.get_splits(self.window_ranges)
#         self.train_samples, self.val_samples, self.train_clouds, self.val_clouds = splits
#
#     def convert(self):
#         scans_dir = os.path.join(self.sequence_path, 'velodyne')
#         os.makedirs(scans_dir, exist_ok=True)
#
#         clouds_dir = os.path.join(self.sequence_path, 'voxel_clouds')
#         os.makedirs(clouds_dir, exist_ok=True)
#
#         clouds = np.sort(np.concatenate([self.train_clouds, self.val_clouds]))
#         clouds = [os.path.join(clouds_dir, cloud) for cloud in clouds]
#
#         val_samples, train_samples = val_samples.astype('S'), train_samples.astype('S')
#         val_clouds, train_clouds = val_clouds.astype('S'), train_clouds.astype('S')
#
#         with h5py.File(os.path.join(sequence_path, 'info.h5'), 'w') as f:
#             f.create_dataset('val', data=val_samples)
#             f.create_dataset('train', data=train_samples)
#             f.create_dataset('val_clouds', data=val_clouds)
#             f.create_dataset('train_clouds', data=train_clouds)
#
#         for cloud, window_range in zip(clouds, self.window_ranges):
#
#             start, end = window_range
#             for j in tqdm(range(start, end + 1), desc=f'Creating scans {start} - {end}'):
#
#                 # Load scans and create global cloud
#                 # Also save the scans to disk
#
#             # Create global cloud
#             # Update voxel masks
