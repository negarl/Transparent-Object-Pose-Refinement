import numpy as np
import os
import torch
import yaml
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from src.pose.model import OptimizationModel
import src.pose.object_pose as object_pose
import open3d as o3d
import src.config as config
import matplotlib.pyplot as plt


def scene_point_clouds(model, rgb_image_path, depth_image_path, scene_number, im_id, isbop=False):
    """
    Calculating the point clouds of the plane and it's transformation matrix (camera to plane)
    :param model: The optimization model
    :param rgb_image_path: Path to the rgb image
    :param depth_image_path: Path to the depth image
    :return:
    points:  point clouds of the plane (Table)
    coefficients:  coefficients of the plane
    T: transformation matrix of the plane
    """

    # if not os.path.exists(f"./detected_plane/{scene_number}"):
    #     os.makedirs(f"./detected_plane/{scene_number}")

    if os.path.exists(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_number}/T_matrix_{im_id}.npy")):
        T = np.load(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_number}/T_matrix_{im_id}.npy"))
        coefficiants = np.load(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_number}/T_coef_{im_id}.npy"))
        return T, coefficiants

    # Plane pcl
    color_raw = o3d.io.read_image(rgb_image_path)
    depth_raw = o3d.io.read_image(depth_image_path)
    depth_scale = 1000.0
    if isbop:
        depth_scale = 10000.0
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=depth_scale)
    scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, model.camera_ins)

    # Indicating with points to consider for plane detection
    indices = np.argwhere(
        np.asarray(scene_pcd.points, dtype=float)[:, 2] <= 1.04)  # TODO_(Done) this would be more compact and faster
    indices = np.reshape(indices, (indices.shape[0]))

    restricted_dist_pcd = scene_pcd.select_by_index(indices)
    coefficients, plane_indices = restricted_dist_pcd.segment_plane(0.005, ransac_n=3, num_iterations=1000)
    plane_pcd = restricted_dist_pcd.select_by_index(plane_indices)
    model.plane_pcd = plane_pcd
    # === plane coefficients to transformation matrix: adapted from https://math.stackexchange.com/a/1957132
    # R: compute basis vectors from n
    if coefficients[2] > 0:  # make sure we're facing camera
        coefficients = [-c for c in coefficients]

    model.plane_coefficients = coefficients
    n = np.array(coefficients[:3]) / np.linalg.norm(coefficients[:3])
    nxy_norm = np.linalg.norm(n[:2])
    R = np.eye(3)
    # - b1 vector orthogonal to n
    R[0, 0] = n[1] / nxy_norm
    R[1, 0] = -n[0] / nxy_norm
    R[2, 0] = 0
    # - b2: vector normal to n and b1 -- n x b1 (-> right-handed)
    R[0, 1] = n[0] * n[2] / nxy_norm
    R[1, 1] = n[1] * n[2] / nxy_norm
    R[2, 1] = -nxy_norm
    # - b3: the normal vector itself
    R[:, 2] = n[:3]

    # t: move -d in direction of n
    points = np.asarray(plane_pcd.points)
    t = -n * coefficients[3]
    centroid_in_plane = (R @ (points[:, :3] - t).T).T.mean(axis=0)
    centroid_in_plane[2] = 0  # only xy
    t += R @ centroid_in_plane

    # compose final matrix
    T = np.eye(4, 4)
    T[:3, :3] = R
    T[:3, 3] = t  # to mm

    np.save(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_number}/T_matrix_{im_id}.npy"), T)
    np.save(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_number}/T_coef_{im_id}.npy"), np.asarray(coefficients))

    return T, np.asarray(coefficients)  # , points, coefficients


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ---- File paths
    import src.config
    # The whole dataset
    dataset_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'scenes/')
    # Camera intrinsics
    camera_intr_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml')
    # Object file
    object_file_path = "../../data/canister_simple.obj"

    scale = 4
    representation = 'q'  # 'q' for [q, t], 'so3' for [so3_log(R), t] or 'se3' for se3_log([R, t])
    lr = 0.04 if representation == 'se3' else 0.02  # Learning rate

    # Reading camera intrinsics
    intrinsics_yaml = yaml.load(open(camera_intr_path, 'r'), Loader=yaml.FullLoader)
    # Object model
    verts, faces_idx, _ = load_obj(object_file_path)
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None].to(device) * 0.7)  # gray, (1, V, 3)
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_idx.verts_idx.to(device)],
        textures=textures
    )

    model = OptimizationModel(mesh, intrinsics_yaml, representation=representation, image_scale=scale).to(device)

    scene_path = os.path.join(dataset_path, "001")
    gt_poses = object_pose.get_gt_pose_camera_frame(os.path.join(scene_path, "poses.yaml"),
                                                    os.path.join(scene_path, "groundtruth_handeye.txt"))

    # For plotting error vs relative pose to camera
    gt_coordinate = [np.asarray(gt_pose) @ np.asarray([[0], [0], [0], [1]]) for gt_pose in gt_poses]
    gt_dis_list = object_pose.calculate_dis(gt_coordinate)

    im_id = 1
    # get ground-truth pose
    T_gt = torch.from_numpy(gt_poses[im_id - 1][None, ...]).to(device)  # Bx4x4 # Why adding one dimension ?
    T_igt = torch.inverse(T_gt)

    # Plane pcl
    color_raw = o3d.io.read_image(os.path.join(scene_path, f"rgb/{im_id:06d}.png"))
    depth_raw = o3d.io.read_image(os.path.join(scene_path, f"depth/{im_id:06d}.png"))
    # import imageio
    # imageio.imread()
    plane_pcd, object_pcd, T = scene_point_clouds(model,
                                                  os.path.join(scene_path, f"rgb/{im_id:06d}.png"),
                                                  os.path.join(scene_path, f"depth/{im_id:06d}.png"))

    plane_T_matrix = torch.from_numpy(T).type(torch.FloatTensor).to(device)
    plane_T_matrix = torch.inverse(plane_T_matrix)
    matrix = plane_T_matrix @ T_gt
    # matrix =  T_gt @  plane_T_matrix
    # points_in_plane = (plane_T_matrix[:3, :3] @ T_gt[..., :3].transpose(2, 1)).transpose(2, 1) \
    #                   + plane_T_matrix[:3, 3]

    mesh = model.meshes
    mesh = model.meshes.verts_list()[0][None, ...]

    points_in_plane = (matrix[:, :3, :3] @ mesh.transpose(2, 1)).transpose(2, 1) \
                      + matrix[:, :3, 3]

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])

    mesh_on = points_in_plane.cpu().numpy()[0]
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(mesh_on)

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(plane_pcd)
    o3d.visualization.draw_geometries([pcd_1, pcd_2, mesh_frame])

    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, model.camera_ins)
    # # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # # o3d.visualization.draw_geometries([pcd])
    #
    # # pcd.voxel_down_sample(voxel_size=0.05)
    # # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    # # pcd.orient_normals_to_align_with_direction([0, 0, -1])
    # # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #
    # # mesh = model.meshes
    # mesh = model.meshes.verts_list()[0][None, ...]
    # mesh_off = (T_gt[:, :3, :3] @ mesh.transpose(-1, -2)).transpose(-1, -2).cpu().numpy()[0] + \
    #            T_gt[:, :3, 3].cpu().numpy()[0]
    # mesh_on = model.meshes.verts_list()[0][None, ...].cpu().numpy()[0]
    # pcd_1 = o3d.geometry.PointCloud()
    # pcd_1.points = o3d.utility.Vector3dVector(mesh_off)
    # pcd_2 = o3d.geometry.PointCloud()
    # pcd_2.points = o3d.utility.Vector3dVector(mesh_on)
    # # o3d.visualization.draw_geometries([pcd_1, pcd_2])
    # # o3d.visualization.draw_geometries([pcd, pcd_1])
    #
    # indices = []
    # for i in range(len(pcd.points)):
    #     if 1.04 > float(pcd.points[i][2]):
    #         indices.append(i)
    # plane_mine = pcd.select_by_index(indices)
    # coefficients, indices2 = plane_mine.segment_plane(0.005, ransac_n=3, num_iterations=1000)
    # o3d.visualization.draw_geometries([pcd, pcd_1])
    # o3d.visualization.draw_geometries([plane_mine])
    # o3d.visualization.draw_geometries([plane_mine.select_by_index(indices2)])

    print("howdie")
