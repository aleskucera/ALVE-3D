import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def plot_pts(pts, marker='go', plot_number=True, **kwargs):
    for i in range(len(pts)):
        plt.plot(pts[i, 0], pts[i, 1], marker, **kwargs)
        if plot_number:
            plt.text(pts[i, 0], pts[i, 1], str(i))


def plot_arrow(source, target, text=None, **kwargs):
    plt.arrow(x=source[0], y=source[1],
              dx=0.9 * (target[0] - source[0]), dy=0.9 * (target[1] - source[1]), **kwargs)
    if text is not None:
        plt.text(x=0.5 * (source[0] + target[0]), y=0.5 * (source[1] + target[1]), s=text)


def query_tree(source, target):
    tree = cKDTree(source)
    dists, idx = tree.query(target)
    return dists, idx


def intersection_demo():
    # np.random.seed(15)
    n1, n2 = 10, 8
    pts1 = 10 * np.random.random((n1, 2))
    pts2 = 10 * np.random.random((n2, 2))

    plt.figure(figsize=(10, 10))

    dists12, idx12 = query_tree(source=pts1, target=pts2)
    dists21, idx21 = query_tree(source=pts2, target=pts1)

    pairs12 = np.vstack([range(len(idx12)), idx12]).T.tolist()
    pairs21 = np.vstack([idx21, range(len(idx21))]).T.tolist()

    intersection = [v for v in pairs12 if v in pairs21]

    common_idx2 = np.asarray(intersection)[:, 0]
    common_idx1 = np.asarray(intersection)[:, 1]

    x_inl = pts1[common_idx1]
    y_inl = pts2[common_idx2]
    assert len(x_inl) == len(y_inl)

    plt.grid()
    plt.axis('equal')
    plot_pts(pts1, marker='go')
    plot_pts(pts2, marker='bo')
    for i in range(len(x_inl)):
        plot_arrow(source=x_inl[i], target=y_inl[i], text='%.2f' % dists12[i], color='b')
    # for i in range(len(idx21)):
    #     plot_arrow(source=pts1[i], target=pts2[idx21[i]], text='%.2f' % dists21[i], color='g')
    plot_pts(pts1[common_idx1], marker='go', plot_number=False, markersize=10)
    plot_pts(pts2[common_idx2], marker='bo', plot_number=False, markersize=10)
    plt.pause(0.01)

    plt.show()

    print('Intersection:')
    print(intersection)


def intersection_dist_threshold_demo(dist_th=0.5):
    # np.random.seed(15)
    n1, n2 = 100, 80
    pts1 = 10 * np.random.random((n1, 2))
    pts2 = 10 * np.random.random((n2, 2))

    plt.figure(figsize=(10, 10))

    dists, ids = query_tree(source=pts1, target=pts2)

    mask = dists <= dist_th

    x_inl = pts1[ids[mask]]
    y_inl = pts2[mask]
    assert len(x_inl) == len(y_inl)

    plt.grid()
    plt.axis('equal')
    plot_pts(pts1, marker='go')
    plot_pts(pts2, marker='bo')
    plot_pts(x_inl, marker='go', plot_number=False, markersize=10)
    plot_pts(y_inl, marker='bo', plot_number=False, markersize=10)
    plt.pause(0.01)

    plt.show()


def main():
    # intersection_demo()
    intersection_dist_threshold_demo()


if __name__ == '__main__':
    main()
