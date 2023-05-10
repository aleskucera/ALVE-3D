import time
import torch


def main():
    import open3d as o3d
    import numpy as np

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set the point cloud data and color
    points = np.random.rand(100, 3)  # Example point cloud data
    colors = np.random.rand(100, 3)  # Example color for each point
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Set the opacity (transparency) of the point cloud
    opacity = 0.5  # Choose the desired opacity value between 0 and 1
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5, 0.5])  # Set all colors to white

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    # size = 4
    # tensor = torch.tensor([1, 2, 3, 4])
    # full_tensor = torch.concatenate((tensor, torch.zeros((size - tensor.shape[0],))))
    # print(full_tensor)


if __name__ == '__main__':
    main()
