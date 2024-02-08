"""
Going through the whole dataset and make the predictions.
Starting file
Written by: Negar Layegh
"""
import argparse
import pandas as pd
import numpy as np
import os
import torch
import yaml
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pose.model import OptimizationModel
import pose.object_pose as object_pose
import json
from collision.environment import scene_point_clouds
from utility.logger import Logger
import config
import trimesh
from matplotlib import pyplot as plt
from itertools import chain
import open3d as o3d

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)  # To check whether we have nan or inf in our gradient calculation


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


parser = argparse.ArgumentParser(description='Tracebot project -- Pose estimation using differentiable rendering')
parser.add_argument('--dataset_path', type=str, default=config.PATHS_DATASET_BOP, help='Path to the whole dataset')
# parser.add_argument('--camera_intr_path', type=str, default=os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml'), help='Camera intrinsics')
parser.add_argument('--objects_path', type=str, default='/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_models/models_eval')
parser.add_argument('--mask_path', type=str, default=config.PATHS_DATASET_BOP, help='Path to the ground truth mask')
parser.add_argument('--scale', type=int, default=2, help='smaller image, faster optimization (but less accurate)')
parser.add_argument('--num_iterations', type=int, default=200, help='Maximum number of iteration for the oprimizer')
parser.add_argument('--num_init_est', type=int, default=1, help='Number of times that the pipeline should be initialized with different init poses')
parser.add_argument('--early_stopping_loss', type=int, default=350, help='If loss is less than this, we stop the optimization')
parser.add_argument('--optimizer_alg_name', type=str, default='adam', help='The name of the optimizer we would like to use')
parser.add_argument('--representation', type=str, default='q', choices=['so3', 'se3', 'q'], help='q for [q, t], so3 for [so3_log(R), t] or se3 for se3_log([R, t])')
parser.add_argument('--loss_func', type=int, default=1, help='Choosing between different loss functions')
parser.add_argument('--mesh_num_samples', type=int, default=500, help='Choosing the number of points for each individual object')

parser.add_argument('--rotation_noise_degree', type=list, default=[1], help='Choosing the rotation degree for adding noise to the ground truth psoe')
parser.add_argument('--rotation_axes', type=str, default='x', help='Choosing the rotation axis for adding noise to the ground truth pose')
parser.add_argument('--transition_noise', type=list, default=[0, 0, 0], help='Choosing the transition noise to add to the ground truth pose')

parser.add_argument('--pose_initialization_method', type=str, default='pose_estimator', help='Choosing between adding noise or using the heuristic. Either noise or heuristic or pose_estimator')

parser.add_argument('--experiment', type=str, default="BOP_loss_7_scenes_refinement_results", help='Making folders for different experiments')
parser.add_argument('--debugging', type=bool, default=False, help='Using the debugging tool')

args = parser.parse_args()

# --------------- Parameter
cudnn_deterministic = True
cudnn_benchmark = False
isbop = True
debug_flag = args.debugging
mesh_num_samples = args.mesh_num_samples
num_init_poses = args.num_init_est

pose_initialization_method = args.pose_initialization_method


# ---- File paths
experiment = args.experiment
dataset_path = args.dataset_path
objects_path = args.objects_path
mask_path = args.mask_path

# ---- Scale: smaller image, faster optimization (but less accurate)
scale = args.scale  # powers of 2

# ---- Noise parameters
rotation_noise_degree_list = args.rotation_noise_degree
rotation_axes_list = args.rotation_axes
trans_noise_list = args.transition_noise

scenes = [1, 2, 3, 7, 9, 10, 11, 12]

lr_list = [
     # 0.015,
    0.02,
#    0.04,
#    0.06,
]
loss_number_list = [
    # 0,
    # 1,
    #  2,
    # 3,
    #4,
    # 5,
    # 6,
    7

]


# get all scene information:
images_per_scene, image_ids_per_scene, gt_poses_per_scene, object_names_per_scene, estimated_poses_per_scene, estimated_obj_names_per_scene = [], [], [], [], [], []

for scene_id in scenes:
    scene_path = os.path.join(dataset_path, f'{scene_id:06d}')

    # get info from poses.yaml
    # gt = ground-truth matrix : model to camera
    gt_img_poses_list, object_names_list, image_ids_list = object_pose.get_gt_pose_camera_frame_bop(os.path.join(scene_path, "scene_gt.json"))
    if pose_initialization_method == 'pose_estimator':
        estimated_img_poses_list, estimated_object_names_list, estimated_image_ids_list = object_pose.get_estimated_pose_initialization(f"/home/negar/Documents/Tracebot/Files/other_methods_results/MegaPose_results/without_refinement/scene_{scene_id}_result_pose_estimation.yaml")

        new_gt_img_poses_list = []
        for img_idx in range(len(object_names_list)):
            img_gt_poses = gt_img_poses_list[img_idx]
            obj_ids_gt = object_names_list[img_idx]
            obj_ids_est = estimated_object_names_list[img_idx]

            objs_idx = [list(obj_ids_gt).index(obj_num) for obj_num in obj_ids_est]
            new_img_poses = [img_gt_poses[idx] for idx in objs_idx]
            new_gt_img_poses_list.append(new_img_poses)

        gt_poses_per_scene.append(new_gt_img_poses_list)
        object_names_per_scene.append(np.unique(np.asarray(list(chain.from_iterable(estimated_object_names_list)))))
        estimated_obj_names_per_scene.append(estimated_object_names_list)
        images_per_scene.append(len(os.listdir(os.path.join(scene_path, 'rgb'))))
        image_ids_per_scene.append(image_ids_list)
        estimated_poses_per_scene.append(estimated_img_poses_list)

    else:
        gt_poses_per_scene.append(gt_img_poses_list)
        object_names_per_scene.append(np.unique(object_names_list))
        images_per_scene.append(len(os.listdir(os.path.join(scene_path, 'rgb'))))
        image_ids_per_scene.append(image_ids_list)

all_object_names = np.unique([name for names in object_names_per_scene for name in names])

# load all meshes that will be needed
cmap = plt.cm.tab20(range(80)) # 20 different colors, two consecutive ones are similar (for two instances)
meshes = {}
sampled_down_meshes = {} # To be used for calculating the sign dist (All the meshes should have the same number of vertices)
meshes_stable_poses = {} # Poses that the object rests on the plane. These poses are used for the initialization

mesh_diameters = {}

f = open(os.path.join(objects_path, 'models_info.json'))
model_info = json.load(f)
f.close()

for oi, object_name in enumerate(all_object_names):
    verts1, faces_idx1 = load_ply(os.path.join(objects_path, f'obj_{object_name:06d}.ply'))
    # Same number of points for each individual object
    mesh_sampled_down = trimesh.load(os.path.join(objects_path, f'obj_{object_name:06d}.ply'))
    mesh_sampled_down.vertices = mesh_sampled_down.vertices #* 0.001
    transforms, probs = trimesh.poses.compute_stable_poses(mesh_sampled_down,  n_samples=5) # Transforms: poses, Prob: probability of that pose happening
    norms = mesh_sampled_down.face_normals
    diameter = model_info[str(object_name)]['diameter']
    mesh_samples = trimesh.sample.sample_surface_even(mesh_sampled_down, 9000)
    faces_idx = torch.from_numpy(np.asarray(mesh_sampled_down.faces[mesh_samples[1]])).type(torch.LongTensor)

    samples = trimesh.sample.sample_surface_even(mesh_sampled_down, mesh_num_samples) # either exactly NUM_samples, or <= NUM_SAMPLES --> pad by random.choice
    samples_norms = norms[samples[1]] # Norms pointing out of the object
    samples_point_norm = np.concatenate((np.asarray(samples[0]), np.asarray(0-samples_norms)), axis=1)
    if samples_point_norm.shape[0] < mesh_num_samples:  # NUM_SAMPLES not equal to mesh_num_samples -> padding
        img_idx = np.random.choice(samples_point_norm.shape[0], mesh_num_samples - samples_point_norm.shape[0])
        samples_point_norm = np.concatenate((samples_point_norm, samples_point_norm[img_idx]), axis=0)

    for ii in range(2):  # two instances per object with different color
        textures = TexturesVertex(verts_features=torch.from_numpy(np.array(cmap[oi*2+ii][:3]))[None, None, :]
                                  .expand(-1, verts1.shape[0], -1).type_as(verts1).to(device)
        )
        print(cmap[oi*2+ii][:3])
        mesh = Meshes(
            verts=[verts1.to(device)],
            faces=[faces_idx1.to(device)],
            textures=textures
        )
        if isbop:
            mesh.scale_verts_(0.001)

        # mesh.scale_verts_(400)
        meshes[f'{object_name}-{ii}'] = mesh
        sampled_down_meshes[f'{object_name}-{ii}'] = samples_point_norm
        mesh_diameters[f'{object_name}-{ii}'] = diameter
        meshes_stable_poses[f'{object_name}-{ii}'] = transforms, probs

# ---- Optimization parameters
optimizor_algorithm_dic = {
    'adam': torch.optim.Adam,
    #'adagrad': torch.optim.Adagrad,
    #'RMSprop': torch.optim.RMSprop,
    #'SGD': torch.optim.SGD,
    #'LBFGS': torch.optim.LBFGS
}

max_num_iterations = args.num_iterations
early_stopping_loss = args.early_stopping_loss
representation = args.representation

# Useful for all scenes
torch.random.manual_seed(0)
torch.backends.cudnn.deterministic = cudnn_deterministic
torch.backends.cudnn.benchmark = cudnn_benchmark
torch.set_default_dtype(torch.float32)
np.random.seed(0)  # note: the rasterizer of PyTorch3d itself is not deterministic (face order is random)
# irrespective of the seed


estimates_bop = ""
bop_results_path = os.path.join(config.PATH_REPO, f"BOP/result/megaposediffRendering-12379-loss7_tless-test_primesense.csv")

if bop_results_path != "" and os.path.exists(bop_results_path):
    open(bop_results_path, 'w').write("")
# Going through each scene number directory
for scene_number, scene_objects, number_of_scene_image, gt_img_poses_list, image_ids_list, estimated_obj_name_list, estimated_poses_list \
        in zip(scenes, object_names_per_scene, images_per_scene, gt_poses_per_scene, image_ids_per_scene, estimated_obj_names_per_scene, estimated_poses_per_scene):

    scene_name = f'{scene_number:06d}'
    print("____________________ scene number : ", scene_name)
    scene_path = os.path.join(dataset_path, scene_name)

    # Optimization model
    model = OptimizationModel(None, None, representation=representation, image_scale=scale,
                              loss_function_num=args.loss_func, isbop=True).to(device)


    # create scene geometry from known meshes
    object_names, object_counts = np.unique(scene_objects, return_counts=True)
    assert object_counts.max() <= 3  # only 2 instances atm
    counter = dict(zip(object_names, [0]*len(object_names)))
    scene_meshes = []
    scene_sampled_mesh = []
    scene_obj_names = [] # For storing the name of the objects
    scene_mesh_diameters = []
    scene_meshes_stable_pose = {}
    obj_id = 0
    for object_name in scene_objects:
        i = counter[object_name]
        counter[object_name] += 1
        mesh = meshes[f'{object_name}-{i}']
        scene_meshes.append(mesh)
        scene_sampled_mesh.append(sampled_down_meshes[f'{object_name}-{i}'])
        scene_obj_names.append(object_name)
        scene_mesh_diameters.append(mesh_diameters[f'{object_name}-{i}'])
        scene_meshes_stable_pose[f'{obj_id}'] = meshes_stable_poses[f'{object_name}-{i}']
        obj_id = obj_id + 1

    # Adding the properties to the model
    model.meshes = scene_meshes
    model.sampled_meshes = scene_sampled_mesh
    model.meshes_name = scene_obj_names
    model.meshes_diameter = scene_mesh_diameters
    model.meshes_stable_pose_dic = scene_meshes_stable_pose

    if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}"))
    # Path to plane's T directory (Check one per scenes)
    if not os.path.exists(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_name}"))
    # Path for saving logs
    if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}"))

    for t_mag in [1]:#range(1, 10):

        for loss_num in loss_number_list:

            model.loss_func_num = loss_num

            if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}")):
                os.mkdir(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}"))
            if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}/loss_num_{loss_num}")):
                os.mkdir(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}/loss_num_{loss_num}"))

            # Path for saving the sdf_images for contour loss
            if not os.path.exists(os.path.join(config.PATH_REPO, f"result/sdf_images/loss_num_{loss_num}/{scene_name}")):
                os.makedirs(
                    os.path.join(config.PATH_REPO, f"result/sdf_images/loss_num_{loss_num}/{scene_name}"))

            for optimizer_name in optimizor_algorithm_dic:

                if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}/{optimizer_name}")):
                    os.mkdir(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}/{optimizer_name}"))
                if not os.path.exists(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}/loss_num_{loss_num}/{optimizer_name}")):
                    os.mkdir(os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}/loss_num_{loss_num}/{optimizer_name}"))

                for lr in lr_list:
                                                                                    #  METHOD_DATASET-test
                    bop_results_path_single = os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}/{optimizer_name}/megaposediffRendering{lr}_tless-test_primesense.csv")

                    if bop_results_path != "" and os.path.exists(bop_results_path_single):
                        open(bop_results_path, 'w').write("")


                    logger = Logger(log_dir=os.path.join(config.PATH_REPO, f"BOP/logs/{scene_name}/loss_num_{loss_num}/{optimizer_name}"),
                                    log_name=f"{scene_name}_opt_{optimizer_name}_lr_{lr}",
                                    reset_num_timesteps=True)

                    print("------- Scene number: ", scene_name)
                    # For each image in the scene

                    # For evaluation
                    best_in_iters = {}
                    scene_iter_value = []
                    # For bop format evaluation
                    estimates_bop_single = ""

                    gt_dis_list = [np.linalg.norm(np.asarray(gt_poses)[:, :3, 3], axis=-1) for gt_poses in gt_img_poses_list]

                    # prepare events
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    # record start
                    start.record()

                    for im_id in image_ids_list: # range(1, number_of_scene_image+1)
                        im_id_index = image_ids_list.index(im_id)
                        im_id = int(im_id)

                        print(f"im {im_id}: optimizing...")
                        # get ground-truth pose
                        T_gt_list = torch.stack([torch.from_numpy(gt_poses).to(device)
                                     for gt_poses in gt_img_poses_list[im_id_index]])  # Bx4x4 , model2camera frame
                        T_igt_list = torch.inverse(T_gt_list)# camera2model

                        # Reading camera intrinsics
                        # Each image have its own intrinsics
                        f = open(os.path.join(scene_path, 'scene_camera.json'))
                        camera_inf_dic = json.load(f)
                        camera_intrinsics = camera_inf_dic[f'{im_id}']['cam_K']
                        model.set_cameras(camera_intrinsics, image_scale=scale)

                        T, coefficiants = scene_point_clouds(model, os.path.join(scene_path, f"rgb/{im_id:06d}.png"),
                                               os.path.join(scene_path, f"depth/{im_id:06d}.png"), scene_name, im_id, isbop=isbop)
                        model.plane_T_matrix = torch.from_numpy(T).type(torch.FloatTensor).to(device)
                        model.plane_coefficients = coefficiants  # Needing for the stable pose, for converting the frames
                        best_metrics, iter_values = object_pose.\
                            scene_optimization(num_init_poses, logger, t_mag, isbop, scene_path, mask_path, scene_name, im_id, model,
                                               T_gt_list, T_igt_list, scale, max_num_iterations, early_stopping_loss, lr,
                                               optimizor_algorithm_dic, optimizer_name, f"BOP/debug/{scene_name}/loss_num_{loss_num}/{optimizer_name}/{lr}", debug_flag, rotation_noise_degree_list, rotation_axes_list,
                                               trans_noise_list, pose_initialization_method, estimated_obj_name_list[im_id_index], estimated_poses_list[im_id_index], scene_objects)
                        scene_iter_value.append(iter_values)

                        duration, conf = 1.0, 0.0

                        obj_idx = -1
                        for object_idx in range(len(scene_obj_names)):
                            if pose_initialization_method == 'pose_estimator':
                                if scene_obj_names[object_idx] not in estimated_obj_name_list[im_id_index]:
                                    continue
                            obj_idx = obj_idx + 1
                            obj_id = scene_obj_names[object_idx]
                            R, t = model.get_R_t(0)
                            # here we need to be in camera frame (m2c)

                            estimates_bop += f"{scene_number:06d},{im_id:06d},{obj_id},{conf:0.3f}, " \
                                             f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['r'])[len(iter_values['r'])-1][obj_idx].transpose(-1, -2).reshape(-1)])}," \
                                             f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['t'])[len(iter_values['t'])-1][obj_idx].reshape(-1) * 1000])}," \
                                             f"{duration:0.3f}\n"

                            estimates_bop_single += f"{scene_number:06d},{im_id:06d},{obj_id},{conf:0.3f}, " \
                                             f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['r'])[len(iter_values['r']) - 1][obj_idx].transpose(-1, -2).reshape(-1)])}," \
                                             f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['t'])[len(iter_values['t']) - 1][obj_idx].reshape(-1) * 1000])}," \
                                             f"{duration:0.3f}\n"

                        logger.record("im_best_ADD", best_metrics['ADD'])
                        logger.dump(step=im_id)

                        for error_name in best_metrics.keys():
                            if error_name not in best_in_iters.keys():
                                best_in_iters.setdefault(error_name)
                                best_in_iters[error_name] = []
                            best_in_iters[error_name].append(best_metrics[error_name])
                        print("End of the image")

                    end.record()
                    torch.cuda.synchronize()
                    # get time between events (in ms)
                    print("____Timing for the whole scene:______", start.elapsed_time(end))

                    with open(bop_results_path_single, 'a') as file:
                        file.write(estimates_bop_single)

                    with open(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}/{optimizer_name}/megapose_{scene_name}_images_dic_lr_{lr}.yaml"), 'w') as fout:
                        json.dump(scene_iter_value, fout)
                    with open(os.path.join(config.PATH_REPO, f"BOP/result/{scene_name}/loss_num_{loss_num}/{optimizer_name}/megapose_{scene_name}_best_iters_lr_{lr}.yaml"), 'w') as fou:
                        json.dump(best_in_iters, fou)

                    logger.close()


with open(bop_results_path, 'a') as file:
    file.write(estimates_bop)