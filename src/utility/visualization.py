import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import torch
import cv2 as cv
import yaml
from pytorch3d.io import load_obj, load_ply
from skimage import img_as_ubyte
from skimage.transform import resize
from pytorch3d.structures import Meshes
import pytorch3d.structures as pyst
from pytorch3d.renderer import TexturesVertex
from src.pose.model import OptimizationModel
import json
import open3d as o3d
# from src.pose.environment import scene_point_clouds
# from src.pose.environment import scene_point_clouds
import src.config as config
import matplotlib as mpl
from matplotlib import pyplot as plt


"""
Installing opencv-python-headless for cv2 because then matplotlib and opencv with have a problem with qt! 
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
cmap = plt.cm.tab20(range(80))


def visualization_progress(model, background_image, r_objs_list, t_objs_list, text=""):
    """

    :param model:
    :param background_image:
    :param r_t_dict:
    :param device:
    :param text:
    :return:
    """
    R_list = torch.stack([torch.from_numpy(np.asarray(r_list)).to(model.device) for r_list in r_objs_list])
    t_list = torch.stack([torch.from_numpy(np.asarray(t_list)).to(model.device) for t_list in t_objs_list])

    image_est = model.ren_vis(meshes_world=model.meshes, R=R_list, T=t_list)
    image_est = image_est.sum(0)
    estimate = image_est[..., -1].detach().cpu().numpy()  # [0, 1]

    # visualization
    vis = background_image[..., :3].copy()
    vis *= 0.5
    vis[..., 2] += estimate * 0.5
    text = ""
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


def result_visualization_single_image(model, source_image_rgb, source_image_mask, scale, r_t_per_iter, best_metrics, visualization_saving_path, image_saving_path):
    """
    Visualizing the detection for a single image
    :param model: Object model, OptimizationModel type
    :param source_image_rgb:
    :param source_image_mask:
    :param scale: #should also save the scale in the files to avoid mistakes !
    :param r_t_per_iter:
    :param best_metrics:
    :param visualization_saving_path:
    :param image_saving_path:
    :return:
    """

    reference_height, reference_width = source_image_rgb.shape[:2]

    if scale != 1:
        reference_width //= scale
        reference_height //= scale
        source_image_rgb = resize(source_image_rgb[..., :3], (reference_height, reference_width))
        source_image_mask = resize(source_image_mask, (reference_height, reference_width))

    visualization_buffer = []
    source_image_rgb[source_image_mask > 0, 1] = source_image_rgb[source_image_mask > 0, 1] * 0.5 \
                                           + source_image_mask[source_image_mask > 0] * 0.5

    t = 0
    for iter_num in range(len(r_t_per_iter["r"])):
        t = iter_num
        visualization = visualization_progress(model, source_image_rgb, r_t_per_iter["r"][iter_num], r_t_per_iter["t"][iter_num], f"Iteration {iter_num + 1:03d}: loss={r_t_per_iter['loss'][iter_num]:0.1f}\n")
        visualization_buffer.append(visualization)#append(img_as_ubyte(visualization))

    metrics_str = ""

    model.init(source_image_mask, torch.stack([torch.from_numpy(np.asarray(r_t_per_iter["best_T"][i]))[None, ...].to(model.device).transpose(-2, -1) for i in range(len(r_t_per_iter["best_T"]))]))
    visualization = visualization_progress(model, source_image_rgb, r_t_per_iter["r"][t], r_t_per_iter["t"][t], f"Best:\n{metrics_str}")
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(visualization_buffer[0])
    plt.axis('off')
    plt.title("initialization")
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.axis('off')
    plt.title(f"best ADI")
    plt.tight_layout()
    plt.savefig(image_saving_path)
    plt.close()

    reference_height, reference_width = source_image_rgb.shape[:2]
    # """
    visualization_buffer = [
        (resize(vis, (reference_height // 16 * 16, reference_width // 16 * 16)) * 255).astype(np.uint8)
        for vis in visualization_buffer]  # note: mp4 requires size to be multiple of macro block
    imageio.mimsave(visualization_saving_path, visualization_buffer) # fps and quality not available anymore , fps=5, quality=10
    # """
    return


def scenes_mesh_reader(objects_path, datasets_path, scenes_id):
    """
    Reading the objects' meshes in one single scene with the same order of the pose.yaml file
    :param objects_path:
    :param datasets_path:
    :param scenes_id:
    :return:
    """

    meshes_dic = {} # dictionary to save the meshes for all the scenes
    scenes_obj_names_dic = {}
    for scene_id in scenes_id:

        scenes_obj_names_dic[f'{scene_id:03d}'] = []
        objects_poses_dic = yaml.load(open(os.path.join(datasets_path, f'{scene_id:03d}/poses.yaml'), 'r'), Loader=yaml.FullLoader)
        for object_dic in objects_poses_dic:
            obj_name = object_dic['id']
            scenes_obj_names_dic[f'{scene_id:03d}'].append(obj_name)
            if obj_name not in meshes_dic.keys():

                verts, faces_idx, _ = load_obj(os.path.join(objects_path, f'{obj_name}/{obj_name}_simple.obj'))
                mesh = Meshes(
                    verts=[verts.to(device)],
                    faces=[faces_idx.verts_idx.to(device)],
                    # textures=textures
                )  # (N, V, F)
                meshes_dic[f'{obj_name}'] = mesh

    return meshes_dic, scenes_obj_names_dic


def bop_scenes_mesh_reader(objects_path, datasets_path, scenes_id):
    """
    Reading the objects' meshes in one single scene with the same order
    :param objects_path:
    :param datasets_path:
    :param scenes_id:
    :return:
    """

    meshes_dic = {} # dictionary to save the meshes for all the scenes
    scenes_obj_names_dic = {}
    for scene_id in scenes_id:

        f = open(os.path.join(datasets_path, f'{scene_id:06d}/scene_gt.json'))
        scene_info = json.load(f)
        obj_ids_list = [object['obj_id'] for object in scene_info[list(scene_info.keys())[0]]]
        scenes_obj_names_dic[f'{scene_id:06d}'] = obj_ids_list

        for object_id in obj_ids_list:
            if object_id not in meshes_dic.keys():

                verts, faces_idx = load_ply(os.path.join(objects_path, f'obj_{object_id:06d}.ply'))
                mesh = Meshes(
                    verts=[verts.to(device)],
                    faces=[faces_idx.to(device)],
                    # textures=textures
                )  # (N, V, F)
                meshes_dic[f'{object_id}'] = mesh

    return meshes_dic, scenes_obj_names_dic


def result_visualization(model, scene_path, predictions_path, meshes_dic, scenes_obj_names_dic, scenes, loss_nums, lr_list, optimizers_list, isbop=False):
    """
    Visualizing the detection for the whole dataset
    :param model: Object model, OptimizationModel type
    :param scene_path: path to the scene images
    :param predictions_path: path to the detection results
    :param scenes: number of the annotated scenes
    :param loss_nums:
    :param lr_list:
    :param optimizers_list:
    :return:
    """

    for scene_id in scenes:
        if isbop:
            scene_id_str = 6
        else:
            scene_id_str = 3
        for loss_num in loss_nums:
            for optimizer_name in optimizers_list:
                for lr in lr_list:
                    with open(os.path.join(predictions_path, f"{str(scene_id).zfill(scene_id_str)}/loss_num_{loss_num}/{optimizer_name}/{str(scene_id).zfill(scene_id_str)}_images_dic_lr_{lr}_trans[0, 0.5, 0]_x_[0].yaml"), 'r') as stream:
                        try:
                            scene_iter_value = yaml.safe_load(stream)
                        except yaml.YAMLError as exc:
                            print(exc)

                    with open(os.path.join(predictions_path, f"{str(scene_id).zfill(scene_id_str)}/loss_num_{loss_num}/{optimizer_name}/{str(scene_id).zfill(scene_id_str)}_best_iters_lr_{lr}_trans[0, 0.5, 0]_x_[0].yaml"), 'r') as st:
                        try:
                            best_in_iters = yaml.safe_load(st)
                        except yaml.YAMLError as exc:
                            print(exc)

                    saving_path = f"{predictions_path}/{str(scene_id).zfill(scene_id_str)}/loss_num_{loss_num}/{optimizer_name}/vis_{lr}"

                    if not os.path.exists(saving_path):
                        os.makedirs(saving_path)

                    meshes = []
                    for obj_id in range(len(scenes_obj_names_dic[f'{str(scene_id).zfill(scene_id_str)}'])):

                        obj_name = scenes_obj_names_dic[f'{str(scene_id).zfill(scene_id_str)}'][obj_id]
                        if isbop:
                            obj_name = str(obj_name)
                        mesh = meshes_dic[obj_name]
                        textures = TexturesVertex(
                            verts_features=torch.from_numpy(np.array(cmap[obj_id * 2 + obj_id][:3]))[None, None, :]
                            .expand(-1, mesh.verts_list()[0].shape[0], -1).type_as(mesh.verts_list()[0]).to(device))
                        mesh.textures = textures
                        if isbop:
                            mesh.scale_verts_(0.001)
                        meshes.append(mesh)


                    # If the number of objects' vertices is different:
                    # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/batching.md
                    # https://pytorch3d.readthedocs.io/en/latest/modules/structures.html#pytorch3d.structures.join_meshes_as_scene

                    model.meshes = pyst.join_meshes_as_batch(meshes)

                    # # Debugging
                    """
                    import open3d as o3d
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[0].cpu().detach().numpy()))
                    pcd1.paint_uniform_color([1, 0, 0])

                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[1].cpu().detach().numpy()))
                    pcd2.paint_uniform_color([0, 1, 0])

                    pcd3 = o3d.geometry.PointCloud()
                    pcd3.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[2].cpu().detach().numpy()))
                    pcd3.paint_uniform_color([0, 0, 1])
                    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], point_show_normal=True)
                    """

                    for image_index, image_name in enumerate(np.sort(os.listdir(os.path.join(scene_path, f"{str(scene_id).zfill(scene_id_str)}/rgb/")))):

                        if image_index == 2:
                            print(image_index)

                        if isbop:
                            im_id = int(image_name[:-4])
                            f = open(os.path.join(scene_path, f'{str(scene_id).zfill(scene_id_str)}/scene_camera.json'))
                            camera_inf_dic = json.load(f)
                            camera_intrinsics = camera_inf_dic[f'{str(im_id)}']['cam_K']
                            model.set_cameras(camera_intrinsics, image_scale=4)
                        else:
                            im_id = int(image_name[3:-4]) - 1

                        reference_rgb = imageio.imread(os.path.join(scene_path, f"{str(scene_id).zfill(scene_id_str)}/rgb/{image_name}"))
                        reference_mask = 0
                        obj_list = scenes_obj_names_dic[f'{str(scene_id).zfill(scene_id_str)}']
                        for obj_num, obj_name in enumerate(obj_list):

                            if isbop:
                                reference_mask += imageio.imread(
                                    os.path.join(scene_path,
                                                 f"{scene_id:06d}/mask/{im_id:06d}_{obj_num:06d}.png"))

                            else:
                                reference_mask += imageio.imread(
                                    os.path.join(scene_path, f"{scene_id:03d}/masks/003_{obj_name}_00{obj_num}_{image_name}"))

                        if isbop:
                            r_t_per_iter = scene_iter_value[image_index]
                        else:
                            r_t_per_iter = scene_iter_value[im_id]
                            # r_t_per_iter = scene_iter_value[0]
                        best_metrics = {}
                        for key in best_in_iters.keys():
                            best_metrics.setdefault(key)
                            if isbop:
                                best_metrics[key] = best_in_iters[key][image_index]
                            else:
                                best_metrics[key] = best_in_iters[key][im_id]
                                # best_metrics[key] = best_in_iters[key][0]
                        if isbop:
                            result_visualization_single_image(model, reference_rgb, reference_mask, 4, r_t_per_iter,
                                                              best_metrics,
                                                              os.path.join(saving_path,
                                                                           f"{im_id:06d}_optimization.mp4"),
                                                              os.path.join(saving_path, f"{im_id:06d}_result.png"))
                        else:
                            result_visualization_single_image(model, reference_rgb, reference_mask, 4, r_t_per_iter, best_metrics,
                                                              os.path.join(saving_path, f"{im_id + 1}_optimization.mp4"), os.path.join(saving_path, f"{im_id + 1}_result.png"))
    return 0


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    isbop = False

    experiment = "loss_num_7_lr_0.02_transition_y_when_breaks" #"loss_num_2_lr_0.02_transition_x_when_breaks" #"loss_num_2_lr_0.02_noise_rotation_when_breaks" #"loss_num_0_three_vari_done"

    if isbop:
        scene_path = config.PATHS_DATASET_BOP
        objects_path = '/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_models/models_eval'
        predictions_path = os.path.join('/home/negar/Documents/Tracebot/Files/negar-layegh-inverse-rendering/BOP/result')

    else:
        scene_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'scenes')
        objects_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'objects')
        predictions_path = os.path.join(f'/home/negar/Documents/Tracebot/Files/negar-layegh-inverse-rendering/result/experiments/{experiment}')

    os.makedirs(predictions_path, exist_ok=True)

    representation = 'q'
    scenes = [2]  # list(range(1, 6))  # only the annotated subset
    loss_num = [7] # from 0 to 6
    lr_list = [
              # 0.015,
          0.02,
        #    0.04,
        #    0.06
    ]
    optimizers_list = ['adam']
    if isbop:
        scenes_meshes_dic, scenes_obj_names_dic = bop_scenes_mesh_reader(objects_path, scene_path, scenes)
        model = OptimizationModel(None, None, representation=representation, image_scale=4, isbop=True).to(device)
    else:
        # Useful for all scenes
        # Reading camera intrinsics
        intrinsics_yaml = yaml.load(open(os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml'), 'r'),
                                    Loader=yaml.FullLoader)
        scenes_meshes_dic, scenes_obj_names_dic = scenes_mesh_reader(objects_path, scene_path, scenes)
        model = OptimizationModel(None, intrinsics_yaml, representation=representation, image_scale=4).to(device)
    result_visualization(model, scene_path, predictions_path, scenes_meshes_dic, scenes_obj_names_dic, scenes, loss_num, lr_list, optimizers_list, isbop=isbop)
