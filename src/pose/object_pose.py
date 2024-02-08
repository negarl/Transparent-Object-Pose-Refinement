import json
import os
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from skimage.transform import resize
#from line_profiler_pycharm import profile
from scipy.spatial.transform.rotation import Rotation
import imageio
from PIL import Image
from src.contour.contour import single_image_edge, img_contour_sdf, imshow, imsavePNG
from torchvision.transforms import GaussianBlur
from scipy.spatial.transform.rotation import Rotation
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def tq_to_m(tq):
    # tq = [tx, ty, tz, qx, qy, qz, qw]
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = tq[:3]
    m[:3, :3] = Rotation.from_quat(tq[3:]).as_matrix()
    return m

def add_noise(t_mag, gt_pose_list, r_error_deg, t_error_list, axes_rotation):
    """
    Adding noise to the ground truth poses 
    :param axes_rotation: Specifies sequence of axes for rotations. Up to 3 characters
                    belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
                    {'x', 'y', 'z'} for extrinsic rotations.
    :param gt_pose_list: list of ground truth pose of the objects' single scene
    :param r_error_deg: float or array_like, shape (N,) or (N, [1 or 2 or 3])
                    Euler angles
    :param t_error_list: transition error list
    :return: Noisy poses list 
    """
    # initialization error: added on top of GT pose
    r_error = Rotation.from_euler(axes_rotation, r_error_deg, degrees=True).as_matrix().astype(np.float32)
    t_error = np.asarray(t_error_list) * float(t_mag / 10)

    new_gt_poses_list = []

    for obj_num in range(len(gt_pose_list)):
        gt_pose = gt_pose_list[obj_num]
        gt_pose[..., :3, :3] = torch.from_numpy(r_error).to(device) @ gt_pose[..., :3, :3]
        gt_pose[..., :3, 3] += torch.from_numpy(t_error).to(device)
        new_gt_poses_list.append(gt_pose)

    return new_gt_poses_list


def init_inplane_translation_obj(model, mask_img, isbop):
    """
    Estimating the position of an object in the plane in camera frame using mask image and camera intrinsics
    """
    # Calculating the transition
    x, y = np.where(mask_img > 0)

    idx_y = int(len(np.unique(y)) / 2)
    pix_y = np.unique(y)[idx_y]

    x_val = x[np.where(y == pix_y)[0]]
    idx_x = int(len(x_val)/2) # index of the x_val
    pix_x = x_val[idx_x]

    p = np.linalg.inv(model.camera_ins.intrinsic_matrix) @ np.asarray(
        [pix_x, pix_y, 1])  # model.camera_ins.intrinsic_matrix[0][0]
    p_norm = p  # / p[-1]
    co = np.asarray(model.plane_coefficients)
    t = (-co[-1]) / (co[0] * p_norm[0] + co[1] * p_norm[1] + co[2])
    xt = p_norm[0] * t
    yt = p_norm[1] * t
    depth_scale = 1000
    return xt / depth_scale, yt / depth_scale, t/depth_scale


# Using Dominik Bauer code
def cluster_matrix(adjacency):
    """
    clustering the stable poses using adjacency matrix
    """
    # bandwidth reduction -> permutation s.t. distance on nonzero entries from the center diagonal is minimized
    from scipy.sparse import csgraph
    r = csgraph.reverse_cuthill_mckee(csgraph.csgraph_from_dense(adjacency), True)
    # via http://raphael.candelier.fr/?blog=Adj2cluster
    # and http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
    # -> results in blocks in the adjacency matrix that correspond with the clusters
    # -> iteratively extend the block while the candidate region contains nonzero elements (i.e. is connected)
    clusters = [[r[0]]]
    for i in range(1, len(r)):
        if np.any(adjacency[clusters[-1], r[i]]):  # ri connected to current cluster? -> add ri to cluster
            clusters[-1].append(r[i])
        else:  # otherwise: start a new cluster with ri
            clusters.append([r[i]])
    # add clustered objects to hypotheses clusters
    adjacencies = []
    for cluster in clusters:
        cluster_adjacency = {}
        for ci in cluster:
            identifier = ci  # obj_ids[ci]
            cluster_adjacency[identifier] = []
            for ci_ in cluster:
                if ci_ == ci:
                    continue
                if adjacency[ci, ci_] == 1:
                    other_identifier = ci_  # obj_ids[ci_]
                    cluster_adjacency[identifier].append(other_identifier)
        adjacencies.append(cluster_adjacency)
    return adjacencies


# with help of : https://nghiaho.com/?page_id=363
def initial_pose_all(num_init_poses, model, ref_mask_list, isbop):
    """
    Random initialization for the poses
    """
    init_batches_poses_list = []
    # non_vis_obj_ids = []
    poses_dict = model.meshes_stable_pose_dic
    clustered_poses_all = []

    for obj_num in range(len(poses_dict)):
        # When an object is not visible in a scene, we should totally ignore it. For BOP
        # if np.sum(ref_mask_list[obj_num]) == 0:
        #     for batch_idx in range(num_init_poses):
        #         if obj_num == 0:
        #             init_batches_poses_list.append([])
        #         init_batches_poses_list[batch_idx].append(torch.eye(4)[None, ...])
        #         clustered_poses_all.append(np.eye(4)[None, ...])
        #         non_vis_obj_ids.append(obj_num)
        #     continue

        # === 1) initial in-plane translation estimate
        xt, yt, zt = init_inplane_translation_obj(model, ref_mask_list[obj_num], isbop)

        # === 2) pre-compute stable
        # --- a) getting the stable poses
        transform, prob = poses_dict[f'{obj_num}']

        # --- b) clustering the stable poses
        cluster_angle = 15
        new_Ts = []
        for T_ in transform:
            T = np.matrix(np.eye(4))
            # use stable z-translation
            T[2, 3] = T_[2, 3]
            # use stable xy-rotation, reject in-plane rotation
            r = Rotation.from_matrix(T_[:3, :3]).as_euler('xyz')
            T[:3, :3] = Rotation.from_euler('xy', r[:2]).as_matrix()
            # use initial estimate of xy-translation
            T[:2, 3] = np.matrix([xt, yt]).reshape(2, 1)
            new_Ts.append(T.copy())

        # angle between all rotations
        qs = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in new_Ts])
        angles = np.dot(qs, qs.T)

        # cluster by angular distance
        cluster_th = np.cos(np.deg2rad(cluster_angle))
        clusters = [list(cluster.keys()) for cluster in cluster_matrix(np.abs(angles) > cluster_th)]

        Ts = []
        for cluster in clusters:
            # a) use first element
            T = new_Ts[cluster[0]]

            # b) weighted average of cluster (assuming equal weights for physics simulation)
            cluster_zs = [new_Ts[c][2, 3] for c in cluster]
            cluster_qs = [np.matrix(qs[c]) for c in cluster]

            # mean quaternion (as described in "Averaging Quaternions" by Markley, Cheng and Crassidis 2007)
            A = np.zeros((4, 4))
            for q in cluster_qs:
                A += np.dot(q.T, q)
            A /= len(cluster)
            eigval, eigvec = np.linalg.eig(A)

            mean_q = eigvec[:, np.argmax(eigval)]
            mean_z = np.mean(cluster_zs)

            T[:3, :3] = Rotation.from_quat(mean_q).as_matrix()
            if isbop:
                T[2, 3] = mean_z/1000
            else:
                T[2, 3] = mean_z

            Ts.append(T)

        clustered_poses_all.append(Ts)
        init_pose_idx = np.random.choice(np.arange(0, len(Ts), 1, dtype=int))
        # tensor_pose = (torch.from_numpy(Ts[init_pose_idx]).to(device)).to(torch.float32)

        for batch_num in range(num_init_poses):
            if obj_num == 0:
                init_batches_poses_list.append([])
            init_pose_idx = np.random.choice(np.arange(0, len(Ts), 1, dtype=int))
            tensor_pose = (torch.from_numpy(Ts[init_pose_idx]).to(device)).to(torch.float32)
            trans_matrix = model.plane_T_matrix @ tensor_pose
            init_batches_poses_list[batch_num].append(trans_matrix)

    # model.non_vis_obj_ids = non_vis_obj_ids
    model.meshes_stable_pose_clustered = clustered_poses_all
    return torch.stack([torch.stack(T) for T in init_batches_poses_list])


def initial_pose_single_obj(model, obj_id):
    """
    Initializing the pose of a single object
    """
    clustered_poses = model.meshes_stable_pose_clustered[obj_id]
    init_pose_idx = np.random.choice(np.arange(0, len(clustered_poses), 1, dtype=int))
    tensor_pose = (torch.from_numpy(clustered_poses[init_pose_idx]).to(device)).to(torch.float32)
    trans_matrix = model.plane_T_matrix @ tensor_pose
    return trans_matrix[None, ...]


def get_gt_pose_camera_frame(object_pose_file_path, camera_pose_file_path):
    """
    Convert poses from world to camera coordinate system for multiple objects
    :param object_pose_file_path: path to the object pose yaml fil
    :param camera_pose_file_path: path to the camera pose yaml file
    :return:
    """
    # object pose (world space): model -> world
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    # List to store the object poses
    objects_gt_poses = []
    # List of objects ids
    objects_id = []
    # Reading the id of each object and calculating the pose of individual objecst
    for object_dic in objects_poses_dic:
        objects_id.append(object_dic['id'])
        obj_pose = np.asarray(object_dic['pose']).reshape(4, 4).astype(np.float32)
        # camera pose: camera -> world
        cam_poses = [tq_to_m([float(v) for v in line.split(" ")[1:]])
                     for line in open(camera_pose_file_path).readlines()]
        # object pose (camera space): model -> world -> camera
        gt_poses = [np.linalg.inv(cam_pose) @ obj_pose for cam_pose in cam_poses]
        objects_gt_poses.append(gt_poses)

    return objects_gt_poses, objects_id


def get_estimated_pose_initialization(object_pose_file_path):
    """
    Convert poses from world to camera coordinate system for multiple objects
    :param object_pose_file_path: path to the object pose yaml fil, m2c frame.
    :return:
    """
    # object pose (world space): model -> world
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    # List to store the object poses
    objects_est_poses_list = []  # shape: (num_images, num_obj_per_image)
    # List of objects ids
    objects_id = []
    image_ids = []
    # Reading the id of each object and calculating the pose of individual objecst
    for image_id in objects_poses_dic:
        objects_gt_poses_image = []  # to store objects per image
        image_ids.append(image_id)
        object_id_per_image = []
        for obj in objects_poses_dic[image_id]:
            object_id_per_image.append(obj['obj_id'])
            obj_pose = np.eye(4).astype(np.float32)
            obj_pose[:3, :3] = np.asarray(obj['R']).reshape((3, 3))
            obj_pose[:3, 3] = np.asarray(obj['t']) / 1000
            # object pose (camera space): model -> camera
            objects_gt_poses_image.append(obj_pose)

        objects_id.append(object_id_per_image)
        objects_est_poses_list.append(objects_gt_poses_image)

    return objects_est_poses_list, objects_id, image_ids


def get_gt_pose_camera_frame_bop(object_pose_file_path):
    """
    Convert poses from world to camera coordinate system for multiple objects
    :param object_pose_file_path: path to the object pose yaml fil, m2c frame.
    :return:
    """
    # object pose (world space): model -> world
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    # List to store the object poses
    objects_gt_poses_list = []  # shape: (num_images, num_obj_per_image)
    # List of objects ids
    objects_id = []
    image_ids = []
    # Reading the id of each object and calculating the pose of individual objecst
    for image_id in objects_poses_dic:
        objects_gt_poses_image = []  # to store objects per image
        object_id_per_image = []
        image_ids.append(image_id)
        for obj in objects_poses_dic[image_id]:
            object_id_per_image.append(obj['obj_id'])
            obj_pose = np.eye(4).astype(np.float32)
            obj_pose[:3, :3] = np.asarray(obj['cam_R_m2c']).reshape((3, 3))
            obj_pose[:3, 3] = np.asarray(obj['cam_t_m2c']) / 1000
            # object pose (camera space): model -> camera
            objects_gt_poses_image.append(obj_pose)

        objects_id.append(object_id_per_image)
        objects_gt_poses_list.append(objects_gt_poses_image)

    return objects_gt_poses_list, np.asarray(objects_id), image_ids


def scene_optimization(num_init_poses, logger, t_mag, isbop, scene_path, mask_path, scene_number, im_id, model, T_gt_list, T_igt_list, scale, max_num_iterations,
                       early_stopping_loss, lr, optimizor_algorithm_dic, optimizer_name, img_debug_name, debug_flag, rotation_noise_degree_list, rotation_axes_list, trans_noise_list, pose_initialization_method, img_objects_names, estimated_poses_list, scene_objects):
    # Get reference images
    reference_rgb = imageio.imread(os.path.join(scene_path, f"rgb/{im_id:06d}.png"))
    # Depending on how many objects in the scene, sum all of them up together
    reference_mask = np.zeros((reference_rgb.shape[0], reference_rgb.shape[1]), dtype=np.float32)
    reference_mask_list = []  # Needed for loss num 7

    ref_mask_list = []  # Masks for the initial pose
    if pose_initialization_method == 'pose_estimator':
        for num_obj in range(len(scene_objects)):
            if scene_objects[num_obj] not in img_objects_names:
                continue

            if isbop:
                obj_mask = imageio.imread(os.path.join(mask_path, f"{scene_number}/mask/{im_id:06d}_{num_obj:06d}.png"))
                if model.loss_func_num == 7 or model.loss_func_num == 8:
                    reference_mask_list.append(obj_mask)
            else:
                obj_mask = imageio.imread(os.path.join(mask_path,
                                                       f"{scene_number}/masks/003_{model.meshes_name[num_obj]}_00{num_obj}_{im_id:06d}.png"))
                if model.loss_func_num == 7 or model.loss_func_num == 8:
                    reference_mask_list.append(obj_mask)

            # Different objects have different pixel values
            reference_mask[obj_mask > 0] = num_obj + 1
            ref_mask_list.append(obj_mask)
    else:
        for num_obj in range(len(T_gt_list)):
            if isbop:
                obj_mask = imageio.imread(os.path.join(mask_path, f"{scene_number}/mask/{im_id:06d}_{num_obj:06d}.png"))
                if model.loss_func_num == 7 or model.loss_func_num == 8:
                    reference_mask_list.append(obj_mask)
            else:
                obj_mask = imageio.imread(os.path.join(mask_path,
                                                       f"{scene_number}/masks/003_{model.meshes_name[num_obj]}_00{num_obj}_{im_id:06d}.png"))
                if model.loss_func_num == 7 or model.loss_func_num == 8:
                    reference_mask_list.append(obj_mask)

            # Different objects have different pixel values
            reference_mask[obj_mask > 0] = num_obj + 1
            ref_mask_list.append(obj_mask)
    if scale != 1:
        reference_height, reference_width = reference_rgb.shape[:2]
        # Create colored instance mask
        reference_mask = reference_mask[..., None].repeat(3, axis=-1)

        reference_width //= scale
        reference_height //= scale
        reference_rgb = resize(reference_rgb[..., :3], (reference_height, reference_width))
        reference_mask = resize(reference_mask, (reference_height, reference_width))

        # Needing for getting the separate masks for objects, for when the method is only to have the loss defined for each obj separately
        if model.loss_func_num == 7 or model.loss_func_num == 8:
            reference_mask_list = [resize(image[..., None].repeat(3, axis=-1), (reference_height, reference_width)) for
                                   image in reference_mask_list]

        # Giving each individual object, different color in the mask image
        for oi in range(1, len(T_gt_list) + 1):
            cmap = plt.cm.tab20(range(20))
            color = np.array(cmap[oi - 1][:3])[None, None, :].repeat(reference_height, axis=0).repeat(reference_width,
                                                                                                      axis=1)
            reference_mask[reference_mask == oi] = color[reference_mask == oi]

    # Add artificial error to get initial pose estimate
    if pose_initialization_method == 'noise':
        T_init_list = add_noise(t_mag, [T_gt.clone() for T_gt in T_gt_list], rotation_noise_degree_list,
                                trans_noise_list, rotation_axes_list)  # Return the list of T_init with noises
        T_init_batch_list = torch.stack(T_init_list)[None, ...]
    elif pose_initialization_method == 'heuristic':
        T_init_batch_list = initial_pose_all(num_init_poses, model, ref_mask_list, isbop)

    elif pose_initialization_method == 'pose_estimator':
        T_init_list = [torch.from_numpy(np.asarray(t)).to(device) for t in estimated_poses_list]
        T_init_batch_list = torch.stack(T_init_list)[None, ...]

    T_init_list_transposed = T_init_batch_list.transpose(-2, -1)

    if model.loss_func_num == 7 or model.loss_func_num == 8:
        model.initial_poses_transposed = T_init_list_transposed
        model.init(reference_mask, T_init_list_transposed,
                   reference_mask_list)  # note: pytorch3d uses [[R,0], [t, 1]] format (-> transposed)
    else:
        model.init(reference_mask, T_init_list_transposed)

    if pose_initialization_method == 'pose_estimator':
        model.pose_initialization_method = pose_initialization_method
        model.scene_objects_names = scene_objects
        model.img_objects_names = img_objects_names
        model.using_estimator = True

    # --------- For loss number 6 and 7, calculate the edge objective
    ref_rgb_no_scene = np.where(reference_mask > 0, reference_rgb, 0)
    im = Image.fromarray(np.uint8(ref_rgb_no_scene * 255))
    ref_gray_no_scene = im.convert("L")
    ref_gray_tensor = None
    #ref_gray_tensor = torch.from_numpy(np.asarray(ref_gray_no_scene))[None, ...].to(device)  # Reference image without the scene in rgb

    # if model.loss_func_num == 6 or model.loss_func_num == 3:
    #
    #     if os.path.exists(f"../result/sdf_images/{scene_number}/sdf_image_{im_id}.pt"):
    #         sdf_image = torch.load(f"../result/sdf_images/{scene_number}/sdf_image_{im_id}.pt")
    #
    #     else:
    #         image_ref = ref_gray_tensor.float() / 255.
    #         gray_img_ref = image_ref[:, None, ...]  # (N, 1, H, W)
    #         os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #
    #         # reference_height, reference_width = reference_rgb.shape[:2]
    #         image_ref_edge = single_image_edge(gray_img_ref, filter="sobel")
    #
    #         # Normalizing #
    #         # image_ref_edge = (image_ref_edge - image_ref_edge.min()) / (image_ref_edge.max() - image_ref_edge.min())
    #         image_ref_edge[torch.where(image_ref[None,] == 0)] = 0  # The output of the single_image_edge doesn't have 0
    #
    #         # Blurring
    #         gaussian = GaussianBlur(5, sigma=(0.1, 2.0))
    #         image_ref_blur = gaussian.forward(image_ref_edge)
    #
    #         # image_ref_blur = image_ref_blur / image_ref_blur.max()
    #         image_ref_blur[torch.where(image_ref_blur < 0.1)] = 0
    #
    #         import cv2
    #         # For making the edge thicker
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #         dilate = cv2.dilate(image_ref_blur.detach().cpu().numpy()[0, 0, ...], kernel, iterations=2)
    #         a = torch.tensor(dilate[None, None, ...]).to(device)
    #
    #         sdf = img_contour_sdf(a)
    #         sdf_image = sdf.to(device)
    #         torch.save(sdf_image, f"../result/sdf_images/{scene_number}/sdf_image_{im_id}.pt")
    #
    #     ref_gray_tensor = sdf_image

    # if model.loss_func_num == 7:
    #
    #     if os.path.exists(
    #             f"../result/sdf_images/loss_num_{model.loss_func_num}/{scene_number}/sdf_image_{im_id}.pt"):
    #         sdf_image_list_tensor = torch.load(
    #             f"../result/sdf_images/loss_num_{model.loss_func_num}/{scene_number}/sdf_image_{im_id}.pt")
    #
    #     else:
    #         image_ref = ref_gray_tensor.float() / 255.
    #         gray_img_ref = image_ref[:, None, ...]  # (N, 1, H, W)
    #         os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #
    #         # reference_height, reference_width = reference_rgb.shape[:2]
    #         image_ref_edge = single_image_edge(gray_img_ref, filter="sobel")
    #
    #         # Normalizing
    #         image_ref_edge = (image_ref_edge - image_ref_edge.min()) / (image_ref_edge.max() - image_ref_edge.min())
    #         image_ref_edge[
    #             torch.where(image_ref[None,] == 0)] = 0  # The output of the single_image_edge doesn't have 0
    #
    #         # Blurring
    #         gaussian = GaussianBlur(5, sigma=(0.1, 2.0))
    #         image_ref_blur = gaussian.forward(image_ref_edge)
    #
    #         image_ref_blur = image_ref_blur / image_ref_blur.max()
    #         image_ref_blur[torch.where(image_ref_blur < 0.1)] = 0
    #
    #         import cv2
    #         # For making the edge thicker
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #         dilate = cv2.dilate(image_ref_blur.detach().cpu().numpy()[0, 0, ...], kernel, iterations=2)
    #         thin_edge_mask_list = np.asarray(reference_mask_list, dtype=np.bool_)[..., 0] * dilate
    #         a = torch.tensor(thin_edge_mask_list[None, ...]).to(device)
    #         sdf_image_list = []
    #         for i in range(len(T_gt_list)):
    #             sdf_image_list.append(img_contour_sdf(a[:, i, :, :][None, ...])[0, 0, :, :].to(device))
    #
    #         sdf_image_list_tensor = torch.stack(sdf_image_list)[:, None, ...]  # (1, N, 3, 3)
    #         torch.save(sdf_image_list_tensor,
    #                    f"../result/sdf_images/loss_num_{model.loss_func_num}/{scene_number}/sdf_image_{im_id}.pt")
    #
    #     ref_gray_tensor = sdf_image_list_tensor

    # ____ Optimization

    # Prepare events, recording time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start
    start.record()

    metrics_batch, metrics_str_batch = [], []
    optimizer_list = []
    # for visualization
    iter_values_batch = []
    for initialization_idx in range(num_init_poses):
        iter_values_batch.append({"r": [], "t": [], "loss": [], "image_id": im_id})
        optimizer_list.append(optimizor_algorithm_dic[optimizer_name]([model.q_list[initialization_idx], model.t_list[initialization_idx]], lr=lr))
        metrics, metrics_str = model.evaluate_progress(T_igt_list.transpose(-2, -1), initialization_idx, isbop=isbop)
        metrics_batch.append(metrics)
        metrics_str_batch.append(metrics_str)

    best_idx = min(range(len(metrics_batch)), key=lambda i: metrics_batch[i]['ADI'])
    best_metrics, best_metrics_str = metrics_batch[best_idx], metrics_str_batch[best_idx]
    best_T_list = T_init_batch_list[best_idx]
    best_R_list, best_t_list = model.get_R_t(best_idx)

    for i in tqdm(range(max_num_iterations)):
        for initialization_idx in range(num_init_poses):

            # Debugging :
            if debug_flag:
                if not os.path.exists(f"../{img_debug_name}/{im_id}/"):
                    os.makedirs(f"../{img_debug_name}/{im_id}/")

            # reinitializing the flagged objects
            if np.sum(model.need_reinitialization[initialization_idx]) > 0 and model.loss_func_num == 7:
                new_q_list = []
                new_t_list = []
                for obj_idx in range(len(model.need_reinitialization[initialization_idx])):
                    if model.need_reinitialization[initialization_idx][obj_idx]:
                        model.need_reinitialization[initialization_idx][obj_idx] = 0
                        if pose_initialization_method == 'pose_estimator' or pose_initialization_method == 'noise':
                            T_init_trans = model.initial_poses_transposed[0, obj_idx, :, :][None, ...]
                        else :
                            T_init = initial_pose_single_obj(model, obj_idx)
                            T_init_trans = T_init.transpose(-2, -1)
                        new_q_list.append(matrix_to_quaternion(T_init_trans[:, :3, :3])[0])
                        new_t_list.append(T_init_trans[:, 3, :3][0])
                    else:
                        new_q_list.append(model.q_list[initialization_idx][obj_idx].detach().cpu().to(device))
                        new_t_list.append(model.t_list[initialization_idx][obj_idx].detach().cpu().to(device))

                model.mid_opt_init(new_q_list, new_t_list, initialization_idx)
                optimizer_list[initialization_idx] = optimizor_algorithm_dic[optimizer_name](model.parameters(), lr=lr)

            optimizer_list[initialization_idx].zero_grad()
            loss, image, signed_dis, diff_rend_loss, signed_dis_loss, contour_loss, contour_diff_img = model(im_id, logger, initialization_idx, ref_gray_tensor, f"{img_debug_name}/{im_id}/init_{initialization_idx}_{i}.png", debug_flag, isbop)

            loss.backward()
            optimizer_list[initialization_idx].step()

            # early stopping
            # activate it again if you want
            # if loss.item() < early_stopping_loss: # How to make it suitable for different mathods, scenes?
            #     break

            # logging
            logger.record(f"loss_value_{im_id}_{initialization_idx}", loss.item())
            if diff_rend_loss is not None:
                logger.record(f"diff_rend_loss_value_{im_id}_{initialization_idx}", diff_rend_loss.item())
            if signed_dis_loss is not None:
                logger.record(f"signed_dis_loss_value_{im_id}_{initialization_idx}", signed_dis_loss.item())
            if contour_loss is not None:
                logger.record(f"contour_loss_value_{im_id}_{initialization_idx}", contour_loss.item())
            logger.record(f"ADD_value_{im_id}_{initialization_idx}", metrics_batch[initialization_idx]['ADD'])
            # logger.dump(step=i)

            # visualization
            if i % 5 == 0:

                metrics, metrics_str = model.evaluate_progress(T_igt_list.transpose(-2, -1), initialization_idx,
                                                               isbop=isbop)

                R_list, t_list = model.get_R_t(initialization_idx)
                iter_values_batch[initialization_idx]["r"].append([(R.to('cpu').detach().numpy()).tolist() for R in R_list])
                iter_values_batch[initialization_idx]["t"].append([(t.to('cpu').detach().numpy()).tolist() for t in t_list])
                iter_values_batch[initialization_idx]["loss"].append(float(loss.to('cpu').detach().numpy()))

                if metrics['ADI'] < best_metrics['ADI']:
                    best_metrics, best_metrics_str = metrics, metrics_str
                    best_R_list, best_t_list = model.get_R_t(initialization_idx)
                    best_T_list = model.get_transform(initialization_idx).transpose(-2, -1)
                    logger.record("best_loss", loss.item())
                    logger.record("ADD_best", metrics['ADD'])

            logger.dump(step=i)

    # record end and synchronize
    end.record()
    torch.cuda.synchronize()
    # get time between events (in ms)
    print("______timing_________: ", start.elapsed_time(end))
    best_idx = min(range(len(metrics_batch)), key=lambda i: metrics_batch[i]['ADI'])
    iter_values = iter_values_batch[best_idx]
    iter_values.setdefault("best_T")
    iter_values['best_T'] = [(best_T.to('cpu').detach().numpy()).tolist() for best_T in best_T_list]
    # print("THE BEST ___________________", np.min(best_dis.cpu().detach().numpy()))
    iter_values["r"].append([(best_R.to('cpu').detach().numpy()).tolist() for best_R in best_R_list])
    iter_values["t"].append([(best_t.to('cpu').detach().numpy()).tolist() for best_t in best_t_list])
    iter_values["loss"].append(0)

    return best_metrics, iter_values
