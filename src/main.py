"""
The pipeline starts from here
Going through the specified scenes within the dataset and saves the predictions in two yaml file,
one for the results of each iteration and the other for the methods best result
"""
import argparse
import numpy as np
import os
import torch
import yaml
from pytorch3d.io import load_obj
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
#from line_profiler import line_profiler

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)  # to check whether we have nan or inf in our gradient calculation

parser = argparse.ArgumentParser(description='Tracebot project -- Pose refienemt using differentiable rendering')
parser.add_argument('--dataset_path', type=str, default=os.path.join(config.PATH_DATASET_TRACEBOT, 'scenes'), help='Path to the whole dataset')
parser.add_argument('--camera_intr_path', type=str, default=os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml'), help='Camera intrinsics')
parser.add_argument('--objects_path', type=str, default=os.path.join(config.PATH_DATASET_TRACEBOT, 'objects'))
parser.add_argument('--mask_path', type=str, default=os.path.join(config.PATH_DATASET_TRACEBOT, 'scenes'), help='Path to the ground truth mask')
parser.add_argument('--scale', type=int, default=2, help='smaller image, faster optimization (but less accurate)')
parser.add_argument('--num_iterations', type=int, default=200, help='Maximum number of iteration for the oprimizer')
parser.add_argument('--num_init_est', type=int, default=1, help='Number of times that the pipeline should be initialized with different init poses')
parser.add_argument('--early_stopping_loss', type=int, default=350, help='If loss is less than this, we stop the optimization')
parser.add_argument('--optimizer_alg_name', type=str, default='adam', help='The name of the optimizer we would like to use')
parser.add_argument('--representation', type=str, default='q', choices=['so3', 'se3', 'q'], help='q for [q, t], so3 for [so3_log(R), t] or se3 for se3_log([R, t])')
parser.add_argument('--loss_func', type=int, default=1, help='Choosing between different loss functions')
parser.add_argument('--mesh_num_samples', type=int, default=500, help='Choosing the number of points for each individual object')

parser.add_argument('--rotation_noise_degree', type=int, default=0, help='Choosing the rotation degree for adding noise to the ground truth psoe')
parser.add_argument('--rotation_axes', type=str, default='x', help='Choosing the rotation axis for adding noise to the ground truth pose')
parser.add_argument('--transition_noise_x', type=float, default=0, help='Choosing the transition noise to add to the ground truth pose')
parser.add_argument('--transition_noise_y', type=float, default=0, help='Choosing the transition noise to add to the ground truth pose')
parser.add_argument('--transition_noise_z', type=float, default=0, help='Choosing the transition noise to add to the ground truth pose')

parser.add_argument('--pose_initialization_method', type=str, default='noise', help='Choosing between adding noise or using the heuristic. Either noise or heuristic or pose_estimator')

parser.add_argument('--experiment', type=str, default="loss_num_7_noise", help='Making folders for different experiments')
parser.add_argument('--debugging', type=bool, default=False, help='Using the debugging tool')

args = parser.parse_args()

# --------------- Parameter
cudnn_deterministic = True
cudnn_benchmark = False
debug_flag = args.debugging
mesh_num_samples = args.mesh_num_samples
num_init_poses = args.num_init_est
isbop = False # defining whether it is a BOP-tless dataset or Tracebot dataset

rotation_noise_degree_list = [args.rotation_noise_degree]
rotation_axes_list = args.rotation_axes
trans_noise_list = []
trans_noise_list.append(args.transition_noise_x)
trans_noise_list.append(args.transition_noise_y)
trans_noise_list.append(args.transition_noise_z)

print(rotation_noise_degree_list)
print(trans_noise_list)
print(rotation_axes_list)

max_num_iterations = args.num_iterations
early_stopping_loss = args.early_stopping_loss
representation = args.representation

pose_initialization_method = args.pose_initialization_method

# ---- File paths
experiment = args.experiment
dataset_path = args.dataset_path
camera_intr_path = args.camera_intr_path
objects_path = args.objects_path
mask_path = args.mask_path

# ---- Scale: smaller image, faster optimization (but less accurate)
scale = args.scale  # powers of 2

scenes = [2] # only the annotated subset

lr_list = [
    # 1e-2,
    # 0.015,
    0.02,
   # 0.04,
#    0.06,
]
loss_number_list = [
     # 0,
    # 1,
     #2,
    # 3,
    #  4,
     # 5,
    # 6,
    7,
     # 8,
    # 9
]


# get all scenes' information:
images_per_scene, gt_poses_per_scene, object_names_per_scene = [], [], []

for scene_id in scenes:
    scene_path = os.path.join(dataset_path, f'{scene_id:03d}')
    # get info from poses.yaml (object poses in world frame) + groundtruth_handeye.txt (camera position in the world
    # frame)
    gt_poses_list, object_names_list = object_pose.get_gt_pose_camera_frame(os.path.join(scene_path, "poses.yaml"),
                                        os.path.join(scene_path, "groundtruth_handeye.txt"))

    gt_poses_per_scene.append(gt_poses_list)
    object_names_per_scene.append(object_names_list)
    images_per_scene.append(len(os.listdir(os.path.join(scene_path, 'rgb'))))
all_object_names = np.unique([name for names in object_names_per_scene for name in names])

# load all meshes that will be needed
cmap = plt.cm.tab20(range(20)) # 20 different colors, two consecutive ones are similar (for two instances)
meshes = {}
sampled_down_meshes = {} # To be used for calculating the sign dist (All the meshes should have the same number of vertices)
meshes_stable_poses = {} # Poses that the object rests on the plane. These poses are used for the initialization

for oi, object_name in enumerate(all_object_names):
    verts, faces_idx, _ = load_obj(os.path.join(objects_path, f'{object_name}/{object_name}_simple.obj'))

    # Same number of points for each individual object
    mesh_sampled_down = trimesh.load(os.path.join(objects_path, f'{object_name}/{object_name}_simple.obj'))
    transforms, probs = trimesh.poses.compute_stable_poses(mesh_sampled_down,  n_samples=5) # Transforms: poses, Prob: probability of that pose happening
    norms = mesh_sampled_down.face_normals
    samples = trimesh.sample.sample_surface_even(mesh_sampled_down, mesh_num_samples) # either exactly NUM_samples, or <= NUM_SAMPLES --> pad by random.choice
    samples_norms = norms[samples[1]] # Norms pointing out of the object
    samples_point_norm = np.concatenate((np.asarray(samples[0]), np.asarray(0-samples_norms)), axis=1)
    if samples_point_norm.shape[0] < mesh_num_samples:  # NUM_SAMPLES not equal to mesh_num_samples -> padding
        idx = np.random.choice(samples_point_norm.shape[0], mesh_num_samples - samples_point_norm.shape[0])
        samples_point_norm = np.concatenate((samples_point_norm, samples_point_norm[idx]), axis=0)

    for ii in range(2):  # two instances per object with different color
        textures = TexturesVertex(verts_features=torch.from_numpy(np.array(cmap[oi*2+ii][:3]))[None, None, :]
                                  .expand(-1, verts.shape[0], -1).type_as(verts).to(device)
        )
        print(cmap[oi*2+ii][:3])
        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces_idx.verts_idx.to(device)],
            textures=textures
        )
        meshes[f'{object_name}-{ii}'] = mesh
        sampled_down_meshes[f'{object_name}-{ii}'] = samples_point_norm
        meshes_stable_poses[f'{object_name}-{ii}'] = transforms, probs

# ---- Optimization parameters
optimizor_algorithm_dic = {
    'adam': torch.optim.Adam,
    #'adagrad': torch.optim.Adagrad,
    #'RMSprop': torch.optim.RMSprop,
    #'SGD': torch.optim.SGD,
    #'LBFGS': torch.optim.LBFGS
}

# Useful for all scenes
torch.random.manual_seed(0)
torch.backends.cudnn.deterministic = cudnn_deterministic
torch.backends.cudnn.benchmark = cudnn_benchmark
torch.set_default_dtype(torch.float32)
np.random.seed(0)  # note: the rasterizer of pytorch3d itself is not deterministic (face order is random)
# irrespective of the seed

# Reading camera intrinsics
intrinsics_yaml = yaml.load(open(camera_intr_path, 'r'), Loader=yaml.FullLoader)

# Optimization model
model = OptimizationModel(None, intrinsics_yaml, representation=representation, image_scale=scale, loss_function_num=args.loss_func).to(device)


# save in BOP format, all the scenes together
estimates_bop = ""

# Going through each scene number directory
for scene_number, scene_objects, number_of_scene_image, gt_poses_list \
        in zip(scenes, object_names_per_scene, images_per_scene, gt_poses_per_scene):
    scene_name = f'{scene_number:03d}'
    print("____________________ scene number : ", scene_name)
    scene_path = os.path.join(dataset_path, scene_name)

    # create scene geometry from known meshes
    object_names, object_counts = np.unique(scene_objects, return_counts=True)
    assert object_counts.max() <= 3  # only 2 instances atm
    counter = dict(zip(object_names, [0]*len(object_names)))
    scene_meshes = []
    scene_sampled_mesh = []
    scene_obj_names = [] # For storing the name of the objects
    scene_meshes_stable_pose = {}
    obj_id = 0
    for object_name in scene_objects:
        i = counter[object_name]
        counter[object_name] += 1
        mesh = meshes[f'{object_name}-{i}']
        scene_meshes.append(mesh)
        scene_sampled_mesh.append(sampled_down_meshes[f'{object_name}-{i}'])
        scene_obj_names.append(object_name)
        scene_meshes_stable_pose[f'{obj_id}'] = meshes_stable_poses[f'{object_name}-{i}']
        obj_id = obj_id + 1

    # Adding the properties to the model
    model.meshes = scene_meshes
    model.sampled_meshes = scene_sampled_mesh
    model.meshes_name = scene_obj_names
    model.meshes_stable_pose_dic = scene_meshes_stable_pose

    if not os.path.exists(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}"))
    # Path to plane's T directory (Check one per scenes)
    if not os.path.exists(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"result/detected_plane/{scene_name}"))
    # Path for saving the edge objective for contour loss
    if not os.path.exists(os.path.join(config.PATH_REPO, f"result/sdf_images/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"result/sdf_images/{scene_name}"))
    # Path for saving logs
    if not os.path.exists(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}")):
        os.makedirs(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}"))

    for t_mag in [1]:#range(1, 10): The magnitude of the translation noise

        for loss_num in loss_number_list:

            # set the loss function number for the model
            model.loss_func_num = loss_num

            if not os.path.exists(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}")):
                os.mkdir(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}"))
            if not os.path.exists(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}/loss_num_{loss_num}")):
                os.mkdir(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}/loss_num_{loss_num}"))

            for optimizer_name in optimizor_algorithm_dic:

                if not os.path.exists(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}")):
                    os.mkdir(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}"))
                if not os.path.exists(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}")):
                    os.mkdir(os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}"))

                for lr in lr_list:

                    bop_results_path = os.path.join(config.PATH_REPO,
                                                    f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}/diffRendering{lr}_Tracebot-test_primesense.csv")

                    if bop_results_path != "" and os.path.exists(bop_results_path):
                        open(bop_results_path, 'w').write("")


                    logger = Logger(log_dir=os.path.join(config.PATH_REPO, f"logs/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}"),
                                    log_name=f"{scene_name}_opt_{optimizer_name}_lr_{lr}",
                                    reset_num_timesteps=True)

                    print("------- Scene number: ", scene_name)
                    # For each image in the scene

                    # For evaluation
                    best_in_iters = {}
                    scene_iter_value = []
                    # Grand-truth distances list
                    gt_dis_list = [np.linalg.norm(np.asarray(gt_poses)[:, :3, 3], axis=-1) for gt_poses in gt_poses_list]

                    # prepare events
                    start = torch.cuda.Event(enable_timing=True)
                    middle = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    # record start
                    start.record()

                    for im_id in range(1, number_of_scene_image+1): #21

                        print(f"im {im_id}: optimizing...")

                        # get ground-truth pose
                        T_gt_list = torch.stack([torch.from_numpy(gt_poses[im_id - 1]).to(device)
                                     for gt_poses in gt_poses_list])  # Bx4x4 , model2camera frame
                        T_igt_list = torch.inverse(T_gt_list)# camera2model

                        T, coefficiants = scene_point_clouds(model, os.path.join(scene_path, f"rgb/{im_id:06d}.png"),
                                               os.path.join(scene_path, f"depth/{im_id:06d}.png"), scene_name, im_id, isbop=isbop)
                        model.plane_T_matrix = torch.from_numpy(T).type(torch.FloatTensor).to(device)
                        model.plane_coefficients = coefficiants # Needing for the stable pose, for converting the frames
                        best_metrics, iter_values = object_pose.\
                            scene_optimization(num_init_poses, logger, t_mag, isbop, scene_path, mask_path, scene_name, im_id, model,
                                               T_gt_list, T_igt_list, scale, max_num_iterations, early_stopping_loss, lr,
                                               optimizor_algorithm_dic, optimizer_name, f"debug/{scene_name}/loss_num_{loss_num}/{optimizer_name}/{lr}", debug_flag, rotation_noise_degree_list, rotation_axes_list,
                                               trans_noise_list, pose_initialization_method, None, None, None)
                        scene_iter_value.append(iter_values)

                        logger.record("im_best_ADD", best_metrics['ADD'])
                        logger.dump(step=im_id)

                        for error_name in best_metrics.keys():
                            if error_name not in best_in_iters.keys():
                                best_in_iters.setdefault(error_name)
                                best_in_iters[error_name] = []
                            best_in_iters[error_name].append(best_metrics[error_name])

                        # Saving in the BOP format
                        # conf, duration = 1.0, 0.0
                        # for obj_idx in range(len(model.meshes_name)):
                        #     obj_id = model.meshes_name[obj_idx]
                        #
                        #     estimates_bop += f"{scene_number:06d},{im_id:06d},{obj_id},{conf:0.3f}, " \
                        #                      f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['r'])[len(iter_values['r']) - 1][obj_idx][0].transpose(-1, -2).reshape(-1)])}," \
                        #                      f"{' '.join([f'{float(v):0.6f}' for v in np.asarray(iter_values['t'])[len(iter_values['t']) - 1][obj_idx][0].reshape(-1) * 1000])}," \
                        #                      f"{duration:0.3f}\n"
                    end.record()
                    torch.cuda.synchronize()
                    # get time between events (in ms)
                    print("____Timing for the whole scene:______", start.elapsed_time(end))

                    with open(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}/{scene_name}_images_dic_lr_{lr}_trans{trans_noise_list}_{rotation_axes_list}_{rotation_noise_degree_list}.yaml"), 'w') as fout:
                        json.dump(scene_iter_value, fout)
                    with open(os.path.join(config.PATH_REPO, f"result/experiments/{experiment}/{scene_name}/loss_num_{loss_num}/{optimizer_name}/{scene_name}_best_iters_lr_{lr}_trans{trans_noise_list}_{rotation_axes_list}_{rotation_noise_degree_list}.yaml"), 'w') as fou:
                        json.dump(best_in_iters, fou)

                    # with open(bop_results_path, 'a') as file:
                    #     file.write(estimates_bop)

                    logger.close()
