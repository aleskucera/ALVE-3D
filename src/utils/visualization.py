import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


# helper function for data visualization
def visualize_imgs(layout='rows', figsize=(20, 10), **images):
    assert layout in ['columns', 'rows']
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        if layout == 'rows':
            plt.subplot(1, n, i + 1)
        elif layout == 'columns':
            plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.tight_layout()
    plt.show()


def visualize_cloud(xyz, color=None):
    assert isinstance(xyz, np.ndarray)
    assert xyz.ndim == 2
    assert xyz.shape[1] == 3  # xyz.shape == (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        assert color.shape == xyz.shape
        color = color / color.max()
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])
