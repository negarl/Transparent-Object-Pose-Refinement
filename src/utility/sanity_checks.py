import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# Util function for loading meshes
import yaml
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
import sys
import os
import open3d as o3d
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from src import config
from src.pose.object_pose import tq_to_m


def sanity_check_complete_mesh():
    """
    Checking whether the mesh is complete and correct
    """
    # add path for demo utils functions

    sys.path.append(os.path.abspath(''))

    from plot_image_grid import image_grid

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR,
                                "/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_models/models_eval/obj_000001.obj")

    # Load obj file
    # mesh = load_objs_as_meshes([obj_filename], device=device)
    verts, faces_idx, _ = load_obj(
        "/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_models/models_eval/obj_000002.obj")
    # verts, faces_idx, _ = load_obj("/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04/objects/container/container_simple.obj")

    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None].to(device) * 0.7)  # gray, (1, V, 3)
    verts = verts * 0.01
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_idx.verts_idx.to(device)],
        textures=textures
    )
    # plt.figure(figsize=(7,7))
    # texture_image=mesh.textures.maps_padded()
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # plt.axis("off");

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    images = renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")

    print("jfg")



def sanity_check_correct_scene(isbop, scene_path, camera_pose_file_path, object_pose_file_path, objects_path):
    """
    Checking whether all the transformations are correct, and we have a perfect view of the scene from camera point of view
    """

    # First:
    # Are the cameras in the scene looking towards the object? Is the object in the correct scale ?
    # Is it inside the circle of the cameras' view?

    camera_matrix = []

    if isbop:
        f = open(camera_pose_file_path)
        camera_poses_dic = json.load(f)
        camera_matrix = np.ones((len(camera_poses_dic), 4, 4))
        keys_array = list(camera_poses_dic.keys())
        for i in range(len(keys_array)):
            key = keys_array[i]
            # print(key)
            cam = np.eye(4)
            cam[:3, :3] = np.asarray(camera_poses_dic[key]['cam_R_w2c']).reshape((3, 3))
            cam[:3, 3] = np.asarray(camera_poses_dic[key]['cam_t_w2c'])
            camera_matrix[i] = cam
    else:
        camera_matrix = [tq_to_m([float(v) for v in line.split(" ")[1:]])
                     for line in open(camera_pose_file_path).readlines()]

    camera_matrix = np.asarray([np.matrix(matr) for matr in camera_matrix])
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    gt = []

    if isbop:
        image_id = "1"
        object_id = 0
        obj = objects_poses_dic[image_id][object_id]
        obj_pose = np.eye(4).astype(np.float32)
        obj_pose[:3, :3] = np.asarray(obj['cam_R_m2c']).reshape((3, 3))
        obj_pose[:3, 3] = np.asarray(obj['cam_t_m2c'])
        gt = np.linalg.inv(camera_matrix[0]) @ obj_pose
        gt[:3, 3] = gt[:3, 3] / 100

    else:
        object_id = 0
        object_dic = objects_poses_dic[object_id]
        gt = np.asarray(object_dic['pose']).reshape(4, 4).astype(np.float32)


    vertex = []

    if isbop:
        object_name = 2
        verts1, faces_idx1 = load_ply(os.path.join(objects_path, f'obj_{object_name:06d}.ply'))
        vertex = verts1
    else:
        object_name = "container"
        verts, faces_idx, _ = load_obj(os.path.join(objects_path, f'{object_name}/{object_name}_simple.obj'))
        vertex = verts


    pcd1 = o3d.geometry.PointCloud()
    if isbop:
        pcd1.normals = o3d.utility.Vector3dVector(
            np.asarray([rotation.T @ np.array([0, 0, 1]) for rotation in camera_matrix[:, :3, :3]]))
        pcd1.points = o3d.utility.Vector3dVector((np.linalg.inv(camera_matrix)[:, :3, 3]))
    else:
        pcd1.normals = o3d.utility.Vector3dVector(
            np.asarray([rotation @ np.array([0, 0, 1]) for rotation in camera_matrix[:, :3, :3]]))
        pcd1.points = o3d.utility.Vector3dVector(((camera_matrix)[:, :3, 3]))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector((gt[:3, :3] @ vertex.numpy().transpose(1,0)).transpose(1,0) + gt[:3, 3])
    o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=True)

    # Second:


if __name__ == '__main__':
    isbop = True

    if isbop:
        scene_path = os.path.join("/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_test_primesense_bop19/test_primesense", '000001')
        camera_pose_file_path = os.path.join(scene_path,
                                             "scene_camera.json")
        object_pose_file_path = os.path.join(scene_path, "scene_gt.json")
        objects_path = "/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_models/models_eval"
        sanity_check_correct_scene(isbop, scene_path, camera_pose_file_path, object_pose_file_path, objects_path)
    else:
        scene_path = os.path.join("/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04", 'scenes')
        camera_pose_file_path = os.path.join("/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04", "groundtruth_handeye.txt")
        object_pose_file_path = os.path.join(scene_path, "001/poses.yaml")
        objects_path = os.path.join("/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04", 'objects')
        sanity_check_correct_scene(isbop, scene_path, camera_pose_file_path, object_pose_file_path, objects_path)

    print("start")

