import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.stats import special_ortho_group


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    # https://github.com/dornik/sporeagent/blob/a95139c47534670f7a47f86adf62d3e488981409/dataset/augmentation.py#L11
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def generate_transform(rot_mag: float = 45.0, trans_mag: float = 0.5):
    """Generate a random SE3 transformation (3, 4) """
    # https://github.com/dornik/sporeagent/blob/a95139c47534670f7a47f86adf62d3e488981409/dataset/augmentation.py#L139

    # Generate rotation
    rand_rot = special_ortho_group.rvs(3)
    axis_angle = Rotation.as_rotvec(Rotation.from_matrix(rand_rot))
    axis_angle /= np.linalg.norm(axis_angle)
    axis_angle *= np.deg2rad(rot_mag)
    rand_rot = Rotation.from_rotvec(axis_angle).as_matrix()

    # Generate translation
    rand_trans = uniform_2_sphere()
    rand_trans *= trans_mag
    rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

    return rand_SE3


if __name__ == '__main__':
    c = generate_transform()
    print(c, "\n")
    e = generate_transform()
    print(e, "\n")
    d = generate_transform(45.0, 0.6)
    print(d)