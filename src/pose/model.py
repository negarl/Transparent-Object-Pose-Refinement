import cv2
import pytorch3d.transforms
import torch
import torch.nn as nn
from src.contour.contour import imshow
from trimesh import Trimesh
import open3d
from src.contour.contour import imsavePNG
from src.pose.object_pose import initial_pose_single_obj

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from pytorch3d.transforms import (
    matrix_to_quaternion, quaternion_to_matrix, so3_log_map, so3_exp_map, se3_log_map, se3_exp_map,
    matrix_to_euler_angles, euler_angles_to_matrix
)
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, BlendParams, SoftSilhouetteShader, SoftGouraudShader, HardPhongShader,
)
from pytorch3d.renderer.mesh.shader import HardFlatShader
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from src.collision import transformations as tra
import numpy as np
import cv2 as cv
import open3d as o3d
from src.contour import contour
# from line_profiler_pycharm import profile


class OptimizationModel(nn.Module):
    """
    Optimization model class. Containing the renderer and the forward method for optimization process
    """
    def __init__(self, meshes, intrinsics_dict, representation='q', image_scale=1, loss_function_num=1, isbop=False):
        super().__init__()
        self.image_ref_sep = None  # Image reference for each object separately (for calculation of the loss 7)
        self.meshes = meshes  # 3d meshes for targeted objects
        self.meshes_name = None  # Name of each mesh
        self.sampled_meshes = None  # 3d mesh of targeted objects sampled down (all to same number of points for calculating sdf loss)
        self.device = device
        self.loss_func_num = loss_function_num   # The number referring to a specific loss calculation
        self.camera_ins = None  # Camera intrinsics
        self.meshes_stable_pose_dic = None  # For initialization, objects stable poses, all
        self.meshes_stable_pose_clustered = None  # Clustered objects' stable poses
        # self.non_vis_obj_ids = None  # To not consider the objects that are not visible from the camera point of view.
        self.meshes_diameter = None  # Only for BOP dataset, needed for evaluation process

        # Plane (Table in this dataset) point clouds and transformation matrix
        self.plane_pcd = None  # Assigned only in the first time of calculation
        self.plane_T_matrix = None  # Plane matrix for calculating the sdf loss
        self.plane_coefficients = None  # Plane coefficients to calculate the initial pose (heuristic)

        # For when we initiate the pipeline with poses from a pose estimator
        self.scene_objects_names = None
        self.img_objects_names = None
        self.using_estimator = False

        # Camera intrinsic
        cam = o3d.camera.PinholeCameraIntrinsic()

        if not isbop:
            cam.intrinsic_matrix = (np.reshape(np.asarray(intrinsics_dict["camera_matrix"]), (3, 3))).tolist()
            cam.height = intrinsics_dict["image_height"]
            cam.width = intrinsics_dict["image_width"]
            self.camera_ins = cam
            # Scale
            # - speed-up rendering
            # - need to adapt intrinsics accordingly
            width, height = intrinsics_dict['image_width'] // image_scale, intrinsics_dict[
                'image_height'] // image_scale
            intrinsics = np.asarray(intrinsics_dict['camera_matrix']).reshape(3, 3)
            intrinsics[:2, :] //= image_scale

            # Camera
            # - "assume that +X points left, and +Y points up and +Z points out from the image plane"
            # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
            # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/renderer_getting_started.md
            # - this is different from, e.g., OpenCV -> inverting focal length achieves the coordinate flip (hacky solution)
            # - see https://github.com/facebookresearch/pytorch3d/issues/522#issuecomment-762793832
            intrinsics[0, 0] *= -1  # Based on the differentiation between the coordinate systems: negative focal length
            intrinsics[1, 1] *= -1
            intrinsics = intrinsics.astype(np.float32)
            cameras = cameras_from_opencv_projection(R=torch.from_numpy(np.eye(4, dtype=np.float32)[None, ...]),
                                                     tvec=torch.from_numpy(np.asarray([[0, 0, 0]]).astype(np.float32)),
                                                     camera_matrix=torch.from_numpy(intrinsics[:3, :3][None, ...]),
                                                     image_size=torch.from_numpy(
                                                         np.asarray([[height, width]]).astype(np.float32)))
            self.cameras = cameras.to(device)

            # SoftRas-style rendering
            # - [faces_per_pixel] faces are blended
            # - [sigma, gamma] controls opacity and sharpness of edges
            # - If [bin_size] and [max_faces_per_bin] are None (=default), coarse-to-fine rasterization is used.
            blend_params = BlendParams(sigma=1e-5, gamma=1e-5, background_color=(
            0.0, 0.0, 0.0))  # vary this, faces_per_pixel etc. to find good value
            soft_raster_settings = RasterizationSettings(
                image_size=(height, width),
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                faces_per_pixel=50,  # I reduced it to 50 from 100
                perspective_correct=True,
            )
            lights = PointLights(device=device, location=((0.0, 0.0, 0.0),), ambient_color=((1.0, 1.0, 1.0),),
                                 diffuse_color=((0.0, 0.0, 0.0),), specular_color=((0.0, 0.0, 0.0),),
                                 )  # at origin = camera center
            self.ren_opt = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=soft_raster_settings
                ),
                # shader=SoftSilhouetteShader(blend_params=blend_params)
                shader=SoftGouraudShader(blend_params=blend_params, device=device, cameras=self.cameras, lights=lights)
                # shader=HardFlatShader(blend_params=blend_params, device=device, cameras=self.cameras, lights=lights)
            )

            # Simple Phong-shaded renderer
            # - faster for visualization

            hard_raster_settings = RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            self.ren_vis = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=hard_raster_settings
                ),
                shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
            )

        # Placeholders for optimization parameters and the reference image
        # - rotation matrix must be orthogonal (i.e., element of SO(3)) - easier to use quaternion
        self.representation = representation
        if self.representation == 'se3':
            self.log_list = None
        elif self.representation == 'so3':
            self.r_list = None
            self.t_list = None
        elif self.representation == 'q':
            self.q_list = None
            self.t_list = None
        self.image_ref = None  # Image mask reference

    def set_cameras(self, intrinsics_dict, image_scale=1):
        """
        Only used for BOP dataset to set up each image separately
        """
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = (np.reshape(np.asarray(intrinsics_dict), (3, 3))).tolist()
        cam.height = 540
        cam.width = 720
        self.camera_ins = cam
        # Scale
        # - speed-up rendering
        # - need to adapt intrinsics accordingly
        width, height = 720 // image_scale, 540 // image_scale
        intrinsics = np.asarray(intrinsics_dict).reshape(3, 3)
        intrinsics[:2, :] //= image_scale

        intrinsics[0, 0] *= -1  # Based on the differentiation between the coordinate systems: negative focal length
        intrinsics[1, 1] *= -1
        intrinsics = intrinsics.astype(np.float32)
        cameras = cameras_from_opencv_projection(R=torch.from_numpy(np.eye(4, dtype=np.float32)[None, ...]),
                                                 tvec=torch.from_numpy(np.asarray([[0, 0, 0]]).astype(np.float32)),
                                                 camera_matrix=torch.from_numpy(intrinsics[:3, :3][None, ...]),
                                                 image_size=torch.from_numpy(
                                                     np.asarray([[height, width]]).astype(np.float32)))
        self.cameras = cameras.to(device)

        # SoftRas-style rendering
        # - [faces_per_pixel] faces are blended
        # - [sigma, gamma] controls opacity and sharpness of edges
        # - If [bin_size] and [max_faces_per_bin] are None (=default), coarse-to-fine rasterization is used.
        blend_params = BlendParams(sigma=1e-5, gamma=1e-5, background_color=(
            0.0, 0.0, 0.0))
        soft_raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=50,
            perspective_correct=True,
        )
        lights = PointLights(device=device, location=((0.0, 0.0, 0.0),), ambient_color=((1.0, 1.0, 1.0),),
                             diffuse_color=((0.0, 0.0, 0.0),), specular_color=((0.0, 0.0, 0.0),),
                             )  # at origin = camera center
        self.ren_opt = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=soft_raster_settings
            ),
            # shader=SoftSilhouetteShader(blend_params=blend_params)
            shader=SoftGouraudShader(blend_params=blend_params, device=device, cameras=self.cameras, lights=lights)
            # shader=HardFlatShader(device=device, cameras=self.cameras)
        )

        # Simple Phong-shaded renderer
        # - faster for visualization

        hard_raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            max_faces_per_bin=None,
        )

        self.ren_vis = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=hard_raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
        )

    def init(self, image_ref, T_init_list, reference_mask_list=None, T_plane=None):
        """
        Initialization for each image of the scene
        """
        self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(device)
        self.image_ref_sep = torch.from_numpy(np.asarray(reference_mask_list).astype(np.float32)).to(device)
        self.need_reinitialization = np.zeros((T_init_list.shape[0], T_init_list.shape[1]))

        if self.representation == 'se3':
            self.log_list = nn.Parameter(se3_log_map(T_init_list))
        elif self.representation == 'so3':
            self.r_list = nn.Parameter(so3_log_map(T_init_list[:, :3, :3]))
            self.t_list = nn.Parameter(T_init_list[:, 3, :3])
        elif self.representation == 'q':  # [q, t] representation
            self.q_list = nn.ParameterList([nn.Parameter(matrix_to_quaternion(T_init[..., :3, :3])) for T_init in T_init_list]) #nn.Parameter(matrix_to_quaternion(T_init_list[..., :3, :3]))
            self.t_list = nn.ParameterList([nn.Parameter(T_init[..., 3, :3]) for T_init in T_init_list])  #nn.Parameter(T_init_list[..., 3, :3])

    def mid_opt_init(self, q_list , t_list, initialization_idx):
        """
        Initialization happens when one object is out of the image space, and we initialize all the objects again
        """
        # if self.representation == 'se3': # not adopted
        #     self.log_list = nn.Parameter(torch.stack([(se3_log_map(T_init)) for T_init in T_init_list]))
        # elif self.representation == 'so3': # not adopted
        #     self.r_list = nn.Parameter(torch.stack([(so3_log_map(T_init[:, :3, :3])) for T_init in T_init_list]))
        #     self.t_list = nn.Parameter(torch.stack([(T_init[:, 3, :3]) for T_init in T_init_list]))
        if self.representation == 'q':  # [q, t] representation
            self.q_list[initialization_idx] = nn.Parameter(torch.stack(q_list))
            self.t_list[initialization_idx] = nn.Parameter(torch.stack(t_list))

    def get_R_t(self, initialization_idx):
        # if self.representation == 'se3':
        #     self.log_list
        #     T = se3_exp_map(self.log)
        #     return T[:, :3, :3], T[:, 3, :3]
        # elif self.representation == 'so3':
        #     return so3_exp_map(self.r), self.t
        if self.representation == 'q':
            return quaternion_to_matrix(self.q_list[initialization_idx]), self.t_list[initialization_idx]

    def get_transform(self, initialization_idx):
        r_list, t_list = self.get_R_t(initialization_idx)
        eye4x4 = torch.eye(4, device=device).unsqueeze(0)
        T_list = eye4x4[:, :, :].expand(r_list.size(0), -1, -1).clone()
        T_list[..., :3, :3] = r_list
        T_list[..., 3, :3] = t_list
        return T_list

    def signed_dis(self, initialization_idx, isbop=False, k=10):
        """
        Calculating the signed distance of an object and the plane (the table)
        :return: two torch arrays with the length of number of points, showing the distance of that point and whether it
        is an intersection point or not
        """

        if self.using_estimator:
            sampled_mesh = []
            for i in range(len(self.scene_objects_names)):
                if self.scene_objects_names[i] not in self.img_objects_names:
                    continue
                sampled_mesh.append(self.sampled_meshes[i])

            points = torch.cat([torch.tensor(sampled_mesh[i][None, ...], dtype=torch.float, device=device) for i in
                 range(len(sampled_mesh))], dim=0)
        else:
            points = torch.cat([torch.tensor(self.sampled_meshes[i][None, ...], dtype=torch.float, device=device) for i in
                                range(len(self.sampled_meshes))],
                               dim=0)  # shape (num of meshes, num point in each obj , 6 (coordinates and norms))

        if isbop:
            points = points / 1000

        estimated_trans_matrixes = self.get_transform(initialization_idx).transpose(-2, -1) # Transposed because of the open3d and pytorch difference

        TOL_CONTACT = 0.01

        # === 1) all objects into plane space
        plane_T_matrix = torch.inverse(self.plane_T_matrix)
        transome_matrixes = plane_T_matrix @ estimated_trans_matrixes

        # For debugging :
        # General debug
        # pcds = []
        # from matplotlib import pyplot as plt
        # cmap = plt.cm.tab20(range(20))
        # for i in range(len(estimated_trans_matrixes)):
        #     point_1 = points[i]
        #     mat = estimated_trans_matrixes[i, ...][None, ...]
        #     point_in_plane_2 = (mat[:, :3, :3] @ point_1[..., :3].transpose(1, 0)).transpose(2, 1) + mat[:, :3,
        #                                                                                              3][:, None, :]
        #     pcd_2 = o3d.geometry.PointCloud()
        #     pcd_2.points = o3d.utility.Vector3dVector(point_in_plane_2.cpu().detach().numpy()[0])
        #     pcd_2.paint_uniform_color(cmap[i * 4, :3])
        #     pcds.append(pcd_2.__copy__())
        # pcds.append(self.plane_pcd)
        # o3d.visualization.draw_geometries(pcds)

        points_in_plane = (transome_matrixes[:, :3, :3] @ points[..., :3].transpose(2, 1)).transpose(2,
                                                                                                     1) + transome_matrixes[
                                                                                                          :, :3, 3][:,
                                                                                                          None, :]

        # === 2) get signed distance to plane
        signed_distance = points_in_plane[..., 2].clone()  # shape (1, N), distance to the plane, is simply the z coordinate of the points in plane frame

        # Indices of other objects that each object should be compared too
        if self.using_estimator:
            others_indices = [list(range(len(self.img_objects_names))) for i in range(len(self.img_objects_names))]
        else:
            others_indices = [list(range(len(self.sampled_meshes))) for i in range(len(self.sampled_meshes))]
        [others_indices[i].remove(i) for i in range(len(others_indices))]

        # === 3) get signed distance to other objects in the scene
        if len(others_indices) > 0:
            # remove scaling (via normalization) for rotation of the normal vectors
            targets_in_plane = points_in_plane.clone()
            normals_in_plane = points[..., 3:6]
            targets_in_plane = torch.cat([targets_in_plane, normals_in_plane], dim=-1)  # The points are in the plane
            # space, but the norms for each individual object does not depend on other object and shows the outside
            # of the object

            batch_signed_distances = []
            batch_support = []
            for b, other_indices in enumerate(others_indices):
                num_other = len(other_indices)

                # get k nearest neighbor points between each of the other objects and the actual object
                distances, nearests = [], []
                for o in other_indices:
                    dist, idx = tra.nearest_neighbor(points_in_plane[o, ..., :3][None, ...],
                                                     points_in_plane[b, ..., :3][None, ...], k=k)
                    near = targets_in_plane[o][idx[0]]  # shape (k, num points, 3)
                    distances.append(dist)
                    nearests.append(near[None, ...])
                if num_other == 0:  # add plane distance instead (doesn't change min)
                    batch_signed_distances.append(signed_distance[b][None, ...])
                    batch_support.append(torch.ones_like(signed_distance[b][None, ...]))
                    continue
                distances = torch.cat(distances, dim=0)  # [num_other] x k x N
                nearests = torch.cat(nearests, dim=0)  # [num_other] x k x N x 6

                # check if query is inside or outside based on surface normal
                surface_normals = nearests[..., 3:6]  # shape (num_other, k, N, 3)
                gradients = nearests[..., :3] - points_in_plane[b][None, :, :3]  # points towards surface (from b to o)
                gradients = gradients / torch.norm(gradients, dim=-1)[..., None]

                # # Debugging
                if torch.any(torch.isnan(nearests[..., :3])):
                    print("nearest :(")
                if torch.any(torch.isnan(gradients)):
                    print("gradient :(")
                if torch.any(torch.isnan(torch.norm(gradients, dim=-1)[..., None])):
                    print("gradient norm :(")

                insides = torch.einsum('okij,okij->oki', surface_normals,
                                       gradients) > 0  # same direction -> inside  #dot-product
                # filter by quorum of votes
                inside = torch.sum(insides, dim=1) > k * 0.8  # shape (num_others, N)

                # get nearest neighbor (in each other object)
                distance, gradient, surface_normal = distances[:, 0, ...], gradients[:, 0, ...], surface_normals[:, 0,
                                                                                                 ...]

                # change sign of distance for points inside
                distance[inside] *= -1

                # take minimum over other points --> minimal SDF overall
                # = the closest outside/farthest inside each point is wrt any environment collider
                if num_other == 1:
                    batch_signed_distances.append(distance[0][None, ...])
                else:
                    distance, closest = distance.min(dim=0)
                    batch_signed_distances.append(distance[None, ...])

            signed_distances = torch.cat(batch_signed_distances, dim=0)

            signed_distance, closest = torch.cat([signed_distance[:, None], signed_distances[:, None]], dim=1).min(
                dim=1)  # the min distance, to which object we have the most collision ? that gives us the answer

        # === 4) derive critical points - allows to determine feasibility and stability
        contacts, intersects = signed_distance.abs() < TOL_CONTACT, signed_distance < -TOL_CONTACT
        # critical_points = torch.cat([contacts[..., None], intersects[..., None], supported[..., None]], dim=-1)

        return signed_distance, intersects

    #@profile
    def forward(self, im_id, logger, initialization_idx, ref_rgb_tensor, image_name_debug, debug_flag, isbop):
        """
        Calculating the forward pass depending on different losses' combination

        differentiable loss : The difference between the rendered mask image and he grand-truth mask.
        Range: [0, double the object pixel size * 2] => in case of no intersection, can lead the object outside
        therefore it reduces the loss to one object pixel size.

        signed_distance loss: Trying to push the objects far from each other. Based on the usage can consider only the
        maximum of the distances or avg or etc.
        Range: [0, inf) => tends to push the objects out

        contour loss :
        """
        # render the silhouette using the estimated pose
        R, t = self.get_R_t(initialization_idx)  # (N, 1, 3, 3), (N, 1, 3)

        binary = True
        as_scene = True

        contour_loss = None
        diff_rend_loss = None
        signed_dis_loss = None
        contour_diff = None
        obj_masks = []
        obj_depth_images = []  # for the loss num 7, single objects rendered depth images
        image_meshe = [] # for bop dataset
        if isbop and self.pose_initialization_method:
            object_idx = [int(np.where(self.scene_objects_names == i)[0]) for i in self.img_objects_names]
            image_meshe = [self.meshes[i] for i in object_idx]

        if self.loss_func_num == 8:
            meshes_faces_num = [0]  # JB's code
            meshes_transformed = []
            if isbop and self.pose_initialization_method:
                for mesh, mesh_r, mesh_t in zip(image_meshe, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)
                    meshes_faces_num.append(meshes_faces_num[-1] + mesh.faces_packed().shape[0])
            else:
                for mesh, mesh_r, mesh_t in zip(self.meshes, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)
                    meshes_faces_num.append(meshes_faces_num[-1] + mesh.faces_packed().shape[0])

            # whole scene mesh
            scene_transformed = join_meshes_as_scene(meshes_transformed)

            # rendering the scene as a whole
            image_est_whole, fragments_est_whole = self.ren_opt(meshes_world=scene_transformed,
                                                                R=torch.eye(3)[None, ...].to(device),
                                                                T=torch.zeros((1, 3)).to(device))

            pix_to_close_face = fragments_est_whole.pix_to_face[..., 0]
            for val_idx, val in enumerate(meshes_faces_num[1:]):
                mesh_mask = (pix_to_close_face >= meshes_faces_num[val_idx]) & (pix_to_close_face < val)
                obj_masks.append(mesh_mask)
        # ----- Loss number 7: needing both scene and separate objects' view rendered images.
        elif self.loss_func_num == 7:
            meshes_transformed = []
            if isbop and self.pose_initialization_method:
                for mesh, mesh_r, mesh_t in zip(image_meshe, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)
            else:
                for mesh, mesh_r, mesh_t in zip(self.meshes, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)
            scene_transformed = join_meshes_as_scene(meshes_transformed)
            image_est_whole, fragments_est_whole = self.ren_opt(meshes_world=scene_transformed,
                                                                R=torch.eye(3)[None, ...].to(device),
                                                                T=torch.zeros((1, 3)).to(device))

            zbuf_whole = fragments_est_whole.zbuf  # (N, h, w, k ) where N in as_scene = 1
            zbuf_whole[zbuf_whole < 0] = torch.inf
            image_depth_est_whole = zbuf_whole.min(dim=-1)[0]
            image_depth_est_whole[torch.isinf(image_depth_est_whole)] = 0

            # unique values of the image_est_whole, later for detecting different objects
            image_unique_whole = torch.round(image_est_whole[0, :, :, 0], decimals=3)

            if isbop and self.pose_initialization_method:
                scene = join_meshes_as_batch(image_meshe)
            else:
                scene = join_meshes_as_batch(self.meshes)
            image_est_separate, fragments_est_separate = self.ren_opt(meshes_world=scene.clone(), R=R,
                                                                      T=t + torch.zeros(t.shape).to(
                                                                          device))

            # Calculating the visible depth image from the scene image and separate image rendered
            for obj_idx in range(len(image_est_separate)):

                # Flag for reinitialization of the specific object because it was shifted out of the image space.
                # Reinitialization happends after the forward function for simplicity and maintaining the computational graph
                if torch.sum(image_est_separate[obj_idx, :, :, 0]) == 0:
                    self.need_reinitialization[initialization_idx][obj_idx] = 1
                obj_value = torch.round(image_est_separate[obj_idx, :, :, 0], decimals=3).unique()
                mask_obj_img = torch.where(image_unique_whole == obj_value[len(obj_value) - 1], 1, 0)
                obj_depth_image = image_depth_est_whole * mask_obj_img[None, ...]
                obj_depth_images.append(obj_depth_image)
                if debug_flag:
                    image_path = image_name_debug[:-4] + f'_obj{obj_idx}_mask_est.png'
                    imsavePNG(obj_depth_image[0, :, :], image_path)

        # for rendering the scene, consider all the objects as a scene and render
        elif as_scene:  # 1 image
            meshes_transformed = []  # transfered meshes to the correct rotation and translation

            if isbop and self.pose_initialization_method:
                for mesh, mesh_r, mesh_t in zip(image_meshe, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)
            else:
                for mesh, mesh_r, mesh_t in zip(self.meshes, R.transpose(-2, -1), t):
                    new_verts_padded = \
                        ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                    mesh = mesh.update_padded(new_verts_padded)
                    meshes_transformed.append(mesh)

            scene_transformed = join_meshes_as_scene(meshes_transformed)

            # fragments have : pix_to_face, zbuf, bary_coords, dists
            image_est, fragments_est = self.ren_opt(meshes_world=scene_transformed,
                                                    R=torch.eye(3)[None, ...].to(device),
                                                    T=torch.zeros((1, 3)).to(device))

            # checking in case of an empty estimate image, flag the object to be reinitialized in the next optimization step
            if torch.sum(image_est[0, :, :, 0]) == 0:
                self.need_reinitialization[initialization_idx] = np.ones(self.need_reinitialization.shape[1])

            # getting depth image from fragments
            zbuf = fragments_est.zbuf  # (N, h, w, k ) where N in as_scene = 1
            zbuf[zbuf < 0] = torch.inf
            image_depth_est = zbuf.min(dim=-1)[0]
            image_depth_est[torch.isinf(image_depth_est)] = 0

        # else:  # N images, not adopted
        #     scene = join_meshes_as_batch(self.meshes)
        #     image_est, fragments_est = self.ren_opt(meshes_world=scene.clone(), R=R[:, 0], T=t[:, 0])
        #     image_est = torch.clip(image_est.sum(dim=0)[None, ...], 0, 1)  # combine simple

        # for debugging
        if debug_flag:
            if self.loss_func_num == 7:
                imsavePNG(image_est_whole[..., 0], image_name_debug)
            else:
                imsavePNG(image_est[:, :, :, 0], image_name_debug)

        # Calculating the signed distance
        signed_dis, intersect_point = self.signed_dis(initialization_idx, isbop=isbop)

        # silhouette loss
        # d's value : [0, double the object pixel size]
        if self.loss_func_num == 7 :
            d = (self.image_ref_sep[..., 0] > 0).float() - image_est_separate[..., 3]
        elif binary:
            d = (self.image_ref[..., 0] > 0).float() - image_est[..., 3]
        else:  # per instance
            d = self.image_ref - image_est[..., :3]

        # _____ Losses calculation

        # differentiable loss: simplest form of the loss combination
        if self.loss_func_num == 0:
            diff_rend_loss = torch.sum(torch.sum(d ** 2))
            loss = diff_rend_loss

        elif self.loss_func_num == 1:
            diff_rend_loss = torch.sum(torch.sum((d[d > 0]) ** 2)) / torch.sum(torch.sum(self.image_ref))
            loss = diff_rend_loss

        elif self.loss_func_num == 2:
            union_mask = ((self.image_ref + image_est[..., :3]) > 0).float()
            loss = torch.sum(torch.sum(d ** 2)) / torch.sum(union_mask)

        elif self.loss_func_num == 3:
            union_mask = ((self.image_ref + image_est[..., :3]) > 0).float()
            diff_rend_loss = torch.sum(torch.sum(d ** 2)) / torch.sum(union_mask)
            signed_dis_loss = torch.max(signed_dis)
            loss = diff_rend_loss + signed_dis_loss

        elif self.loss_func_num == 4:
            union_mask = ((self.image_ref + image_est[..., :3]) > 0).float()
            diff_rend_loss = torch.sum(torch.sum(d ** 2)) / torch.sum(union_mask)
            signed_dis_loss = torch.max(signed_dis)
            loss = diff_rend_loss + signed_dis_loss * 0.001

        elif self.loss_func_num == 5:

            union_mask = ((self.image_ref + image_est[..., :3]) > 0).float()
            diff_rend_loss = torch.sum(torch.sum(d ** 2)) / torch.sum(union_mask)
            signed_dis_loss = torch.max(signed_dis)
            loss = diff_rend_loss + signed_dis_loss * 0.0001

        elif self.loss_func_num == 6:
            contour_diff = contour.contour_loss(image_depth_est, ref_rgb_tensor, image_name_debug, debug_flag)
            diff_rend_loss = torch.sum(torch.sum(d ** 2))
            signed_dis_loss = torch.max(signed_dis)
            contour_loss = torch.sum(contour_diff[contour_diff > 0]) / (contour_diff.shape[-2] * contour_diff.shape[-1])

            loss = diff_rend_loss + signed_dis_loss + contour_loss * 0.0001  # 0.00001

        elif self.loss_func_num == 7:
            loss = 0
            diff_rend_loss_all = 0
            signed_dis_loss_all = 0
            for obj_idx in range(len(image_est_separate)):
                union_mask = ((self.image_ref_sep[obj_idx] + image_est_separate[obj_idx][..., :3]) > 0).float()
                diff_rend_loss = torch.sum(torch.sum(d[obj_idx] ** 2)) / torch.sum(union_mask)
                signed_dis_loss = torch.max(signed_dis[obj_idx])
                loss += diff_rend_loss + signed_dis_loss * 0.001
            loss = loss / len(image_est_separate)

        elif self.loss_func_num == 8:

            loss = 0
            diff_rend_loss_all = 0
            signed_dis_loss_all = 0
            # diff_rend_loss = torch.zeros(len(self.image_ref_sep)).to(device)
            for mask_idx, ref_mask in enumerate(self.image_ref_sep):
                image_unique_mask = torch.where(
                    obj_masks[mask_idx] > 0, image_est_whole[..., 3], 0)

                if debug_flag:
                    image_path = image_name_debug[:-4] + f'_obj{mask_idx}_mask_est.png'
                    imsavePNG(image_unique_mask, image_path)

                union = ((ref_mask[..., 0] + image_unique_mask[0]) > 0).float()
                d = (ref_mask[..., 0] > 0).float() - image_unique_mask[0]
                diff_rend_loss = torch.sum(d ** 2) / torch.sum(union)  # Corresponds to 1 - IoU
                signed_dis_loss = torch.max(signed_dis[mask_idx])
                loss += diff_rend_loss + signed_dis_loss * 0.001

            loss = loss / len(self.image_ref_sep)
        else:
            raise ValueError()

        return loss, None, None, diff_rend_loss, signed_dis_loss, contour_loss, contour_diff  # signed_dis

    def evaluate_progress(self, T_igt_list, initialization_idx, isbop=None):
        # note: we use the [[R, t], [0, 1]] convention here -> transpose all matrices
        T_est_list = self.get_transform(initialization_idx).transpose(-2, -1)
        T_res_list = T_igt_list.transpose(-2, -1) @ T_est_list # [T_igt_list[i].transpose(-2, -1) @ T_est_list[i] for i in range(len(T_igt_list))]  # T_est @ T_igt

        # metrics_list = []
        # metrics_str_list = []
        metrics_dict = {
            'R_iso': [],  # [deg] error between GT and estimated rotation
            't_iso': [],  # [mm] error between GT and estimated translation
            'ADD_abs': [],  # [mm] average distance between model points
            'ADI_abs': [],  # -//- nearest model points
            'ADD': [],  # [%] of model diameter
            'ADI': [],  # -//- nearest model points
        }

        for i in range(len(T_res_list)): # BOP: range(len(T_res_list)) Tracebot:range(len(self.meshes))
            T_res = T_res_list[i][None, ...]

            # isometric errors
            R_trace = T_res[:, 0, 0] + T_res[:, 1, 1] + T_res[:, 2, 2]  # note: torch.trace only supports 2D matrices
            R_iso = torch.rad2deg(torch.arccos(torch.clamp(0.5 * (R_trace - 1), min=-1.0, max=1.0)))
            metrics_dict["R_iso"].append(float(R_iso))
            t_iso = torch.norm(T_res[:, :3, 3])
            metrics_dict["t_iso"].append(float(t_iso))
            # ADD/ADI error

            if isbop:
                diameters = self.meshes_diameter[i]
                mesh_pytorch = self.meshes[i]
                # create from numpy arrays
                d_mesh = open3d.geometry.TriangleMesh(
                    vertices=open3d.utility.Vector3dVector(
                        mesh_pytorch.verts_list()[0].cpu().detach().numpy().copy()),
                    triangles=open3d.utility.Vector3iVector(
                        mesh_pytorch.faces_list()[0].cpu().detach().numpy().copy()))
                simple = d_mesh.simplify_quadric_decimation(
                    int(9000))
                mesh = torch.from_numpy(np.asarray(simple.vertices))[None, ...].type(torch.FloatTensor).to(device)
            else:
                mesh = self.meshes[i].verts_list()[0][None, ...]
                diameters = torch.sqrt(square_distance(mesh, mesh).max(dim=-1)[0]).max(dim=-1)[0]

            mesh_off = (T_res[:, :3, :3] @ mesh.transpose(-1, -2)).transpose(-1, -2) + T_res[:, :3, 3][:, None, :]
            dist_add = torch.norm(mesh - mesh_off, p=2, dim=-1).mean(dim=-1)
            dist_adi = torch.sqrt(square_distance(mesh, mesh_off)).min(dim=-1)[0].mean(dim=-1)
            metrics_dict["ADD_abs"].append(float(dist_add) * 1000)
            metrics_dict["ADI_abs"].append(float(dist_adi) * 1000)
            metrics_dict["ADD"].append(float(dist_add / diameters) * 100)
            metrics_dict["ADI"].append(float(dist_adi / diameters) * 100)
        metrics = {
            'R_iso': np.mean(np.asarray(metrics_dict["R_iso"])),  # [deg] error between GT and estimated rotation
            't_iso': np.mean(np.asarray(metrics_dict["t_iso"])),  # [mm] error between GT and estimated translation
            'ADD_abs': np.mean(np.asarray(metrics_dict["ADD_abs"])),  # [mm] average distance between model points
            'ADI_abs': np.mean(np.asarray(metrics_dict["ADI_abs"])),  # -//- nearest model points
            'ADD': np.mean(np.asarray(metrics_dict["ADD"])),  # [%] of model diameter
            'ADI': np.mean(np.asarray(metrics_dict["ADI"])),  # -//- nearest model points
        }
        metrics_str = f"R={metrics['R_iso']:0.1f}deg, t={metrics['t_iso']:0.1f}mm\n" \
                      f"ADD={metrics['ADD_abs']:0.1f}mm ({metrics['ADD']:0.1f}%)\n" \
                      f"ADI={metrics['ADI_abs']:0.1f}mm ({metrics['ADI']:0.1f}%)"

        return metrics, metrics_str

    def visualize_progress(self, background, text=""):
        # estimate
        R, t = self.get_R_t()
        image_est = self.ren_vis(meshes_world=self.meshes, R=R, T=t)
        estimate = image_est[0, ..., -1].detach().cpu().numpy()  # [0, 1]
        silhouette = estimate > 0

        # visualization
        vis = background[..., :3].copy()
        # add estimated silhouette
        vis *= 0.5
        vis[..., 2] += estimate * 0.5
        # # add estimated contour
        # contour, _ = cv.findContours(np.uint8(silhouette[..., -1] > 0), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
        # vis = cv.drawContours(vis, contour, -1, (0, 0, 255), 1, lineType=cv.LINE_AA)
        if text != "":
            # add text
            rect = cv.rectangle((vis * 255).astype(np.uint8), (0, 0), (250, 100), (167, 168, 168), -1)
            vis = ((vis * 0.5 + rect / 255 * 0.5) * 255).astype(np.uint8)
            font_scale, font_color, font_thickness = 0.5, (0, 0, 0), 1
            x0, y0 = 25, 25
            for i, line in enumerate(text.split('\n')):
                y = int(y0 + i * cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] * 1.5)
                vis = cv.putText(vis, line, (x0, y),
                                 cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv.LINE_AA)
        return vis


def square_distance(pcd1, pcd2):
    # via https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/13
    r_xyz1 = torch.sum(pcd1 * pcd1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(pcd2 * pcd2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(pcd1, pcd2.permute(0, 2, 1))  # (B,M,N)
    return r_xyz1 - 2 * mul + r_xyz2.permute(0, 2, 1)
