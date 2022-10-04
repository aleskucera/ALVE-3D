import numpy as np

from src.datasets.semantic import SemanticUSL


def demo(n_runs=1):
    from src.active_learning.utils import visualize_imgs, visualize_cloud

    ds = SemanticUSL()
    ds_trav = SemanticUSL(output='traversability')
    ds_flex = SemanticUSL(output='flexibility')

    for _ in range(n_runs):
        idx = np.random.choice(range(len(ds)))

        data, label = ds[idx]
        label_trav = ds_trav[idx][1]
        label_flex = ds_flex[idx][1]

        depth_img = data[-1]

        power = 16
        depth_img_vis = np.copy(depth_img)
        depth_img_vis[depth_img_vis > 0] = depth_img_vis[depth_img_vis > 0] ** (1 / power)
        depth_img_vis[depth_img_vis > 0] = \
            (depth_img_vis[depth_img_vis > 0] - depth_img_vis[depth_img_vis > 0].min()) / \
            (depth_img_vis[depth_img_vis > 0].max() - depth_img_vis[depth_img_vis > 0].min())

        color = ds.label_to_color(label)
        color_trav = ds_trav.label_to_color(label_trav)
        color_flex = ds_flex.label_to_color(label_flex)

        visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color.reshape((-1, 3)))
        # visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color_trav.reshape((-1, 3)))
        # visualize_cloud(xyz=data[:3].reshape((3, -1)).T, color=color_flex.reshape((-1, 3)))

        visualize_imgs(range_image=depth_img_vis,
                       segmentation=color,
                       traversability=color_trav,
                       flexibility=color_flex,
                       layout='columns')


def main():
    demo(5)


if __name__ == '__main__':
    main()
