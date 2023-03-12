import numpy as np


def project_points(points: np.ndarray, H: int, W: int, fov_up: float, fov_down: float) -> dict:
    """ Project a point cloud to a depth image

    :param points: point cloud
    :param H: height of the depth image
    :param W: width of the depth image
    :param fov_up: field of view up
    :param fov_down: field of view down
    :return: depth image, remission image, x image, y image, mask image
    """

    # Project to 2D
    proj_x, proj_y, r = proj(points, H, W, fov_up, fov_down)

    # Initialize projection images
    proj_idx = np.full((H, W), -1, dtype=np.int32)
    proj_xyz = np.zeros((H, W, 3), dtype=np.float32)
    proj_depth = np.full((H, W), -1, dtype=np.float32)

    # Order in decreasing depth so that the first point in each pixel is the closest one
    order = np.argsort(-r)
    indices = np.arange(len(points))

    # Sort all arrays by depth
    r = r[order]
    points = points[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]
    indices = indices[order]

    # Fill in projection matrix
    proj_depth[proj_y, proj_x] = r
    proj_mask = proj_depth > 0
    proj_xyz[proj_y, proj_x] = points
    proj_idx[proj_y, proj_x] = indices

    return {'depth': proj_depth, 'xyz': proj_xyz, 'idx': proj_idx, 'mask': proj_mask}


def cart2sph(points: np.ndarray) -> tuple:
    """ Convert cartesian coordinates to spherical coordinates

    :param points: cartesian coordinates
    :return: spherical coordinates
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    elev = np.arctan2(y, x)
    azim = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return r, elev, azim


def proj(points: np.ndarray, H: int, W: int, fov_up: float, fov_down: float) -> tuple:
    """ Project a point cloud to a 2D image

    :param points: point cloud
    :param H: height of the image
    :param W: width of the image
    :param fov_up: field of view up
    :param fov_down: field of view down
    :return: 2D image
    """

    # Convert euclidean coordinates to spherical coordinates
    r, elev, azim = cart2sph(points)

    # Convert FOV to radian
    fov_up = to_rad(fov_up)
    fov_down = to_rad(fov_down)
    total_fov = abs(fov_down) + abs(fov_up)

    # project to 2D
    proj_x = 0.5 * (elev / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (azim + abs(fov_down)) / total_fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= W  # in [0.0, W]
    proj_y *= H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = to_pixels(proj_x, W)
    proj_y = to_pixels(proj_y, H)

    return proj_x, proj_y, r


def to_rad(angle: float) -> float:
    """ Convert an angle from degree to radian

    :param angle: angle in degree
    :return: angle in radian
    """

    return angle / 180 * np.pi


def to_pixels(coord: np.ndarray, size: int) -> np.ndarray:
    """ Convert a coordinate to pixel

    :param coord: coordinate
    :param size: dimension of the image
    :return: pixel
    """

    coord = np.floor(coord)
    coord = np.minimum(size - 1, coord)
    coord = np.maximum(0, coord).astype(np.int32)
    return coord
