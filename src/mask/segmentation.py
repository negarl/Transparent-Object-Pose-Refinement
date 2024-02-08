import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from src.collision.plane_detector import PlaneDetector


class Segmentation:

    def __init__(self, intrinsics_dict, image_scale=1, plane_threshold=0.005):
        self.width, self.height = intrinsics_dict['width'] // image_scale, intrinsics_dict['height'] // image_scale
        self.intrinsics = np.asarray(intrinsics_dict['intrinsics']).reshape(3, 3)
        self.intrinsics[:2, :] //= image_scale
        self.image_scale = image_scale

        self.det = PlaneDetector(self.width, self.height, self.intrinsics, distance_threshold=plane_threshold)

    def _plane_popout(self, c, d, max_dist=5e2, get_plane=False):
        # plane fitting
        T, plane, scene, cloud, indices = self.det.detect(c, d, max_dist)
        keep = plane if get_plane else scene
        # remove outliers
        keep, _ = keep.remove_radius_outlier(nb_points=100//self.image_scale**2, radius=0.01)
        # unproject points
        coords = np.asarray(keep.points)
        coords[..., 0] = coords[..., 0] / coords[..., 2] * self.intrinsics[0, 0] + self.intrinsics[0, 2]
        coords[..., 1] = coords[..., 1] / coords[..., 2] * self.intrinsics[1, 1] + self.intrinsics[1, 2]
        # clip at max_dist
        coords = coords[coords[..., 2] < max_dist * self.det.to_meters]
        # to image-space coordinates
        coords_upper = np.ceil(coords).astype(np.int32)
        coords_lower = np.floor(coords).astype(np.int32)
        coords = np.vstack([coords_lower, coords_upper])
        coords[..., 0] = np.clip(coords[..., 0], 0, self.width - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, self.height - 1)
        # create mask and fill holes
        mask = np.zeros((self.height, self.width))
        mask[coords[..., 1], coords[..., 0]] = 1
        mask = binary_fill_holes(mask)
        return mask, T

    def plane(self, c, d, max_dist=5e2):
        return self._plane_popout(c, d, max_dist=max_dist, get_plane=True)

    def popout(self, c, d, max_dist=5e2):
        return self._plane_popout(c, d, max_dist=max_dist, get_plane=False)

    def difference(self, i1, i2, max_dist=5e2, min_diff=10):
        if len(i1.shape) == 3:  # rgb-based
            m1 = i1.astype(np.float32).mean(axis=-1) / 255
            m2 = i2.astype(np.float32).mean(axis=-1) / 255
            return np.abs(m1 - m2) > min_diff
        else:  # depth-based
            # - difference (but at least min_diff)
            diff = np.abs(i1 - i2)
            diff[diff < min_diff] = 0
            # - remove invalid points
            # closest depth (ignoring invalid points, i.e., depth=0)
            combined = np.dstack([i1, i2])
            combined[..., 0][i1 == 0] = np.infty
            combined[..., 1][i2 == 0] = np.infty
            closest = np.min(combined, axis=-1)
            closest[closest == np.infty] = 0
            # depth shadows and other invalids
            diff[closest == 0] = 0
            # background
            diff[closest > max_dist] = 0
            return diff > 0
