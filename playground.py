import numpy as np

poses = np.array([], dtype=np.float32).reshape((0, 4, 4))

pose_1 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]]], dtype=np.float32)
poses = np.concatenate((poses, pose_1))

pose_2 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]]], dtype=np.float32)
poses = np.concatenate((poses, pose_2))
