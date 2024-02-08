import cv2
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np
import trimesh
from easydict import EasyDict as edict
import sys
import os
import json
from tqdm import tqdm
import pyrender

cwd = os.getcwd()
bop_toolkit_path = "/home/negar/Documents/Tracebot/Files/bop_toolkit-master"
sys.path.append(bop_toolkit_path)
sys.path.append(os.path.join(bop_toolkit_path, "bop_toolkit_lib"))

import bop_toolkit_lib.inout as bop_inout
import bop_toolkit_lib.visibility as bop_vis


groundtruth_to_pyrender = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])


def project_mesh_to_2d(model_path, obj_pose, intrinsic, cam_pose=np.eye(4)):
    # --- PyRender scene setup ------------------------------------------------
    scene = pyrender.Scene(bg_color=[0, 0, 0])

    seg_node_map = {}
    # Add model mesh

    model = trimesh.load(model_path)
    pyr_mesh = pyrender.Mesh.from_trimesh(model, smooth=False)
    scene.add(pyr_mesh, pose=obj_pose)
    nm = pyrender.Node(mesh=pyr_mesh)
    scene.add_node(nm)

    # Add camera
    camera = pyrender.camera.IntrinsicsCamera(intrinsic.fx,
                                              intrinsic.fy,
                                              intrinsic.cx,
                                              intrinsic.cy)

    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)
    nl = pyrender.Node(matrix=np.eye(4))
    scene.add_node(nl)

    # --- Rendering -----------------------------------------------------------
    r = pyrender.OffscreenRenderer(intrinsic.width, intrinsic.height)
    cam_pose = cam_pose.dot(groundtruth_to_pyrender)
    # Render
    scene.set_pose(nc, pose=cam_pose)
    scene.set_pose(nl, pose=cam_pose)

    # Adding light
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=cam_pose)

    img, depth = r.render(
        scene,
        flags=pyrender.RenderFlags.SKIP_CULL_FACES |
              pyrender.RenderFlags.FLAT)

    return img, depth


def transform_cam_poses_to_matrix_form(df_cam_poses_world):
    camera_poses_in_world_coords = {}

    for i, row in df_cam_poses_world.iterrows():
        trans = np.asarray([[row[1], row[2], row[3], 1.]]).T
        rot3x3 = np.asarray(Rotation.from_quat([row[4], row[5], row[6], row[7]]).as_matrix())
        rot_3x4 = np.vstack((rot3x3, np.zeros((1, rot3x3.shape[0]))))  # projection matrix
        camera_poses_in_world_coords[i] = np.hstack((rot_3x4, trans))

    return camera_poses_in_world_coords


def transform_object_from_world_to_camera(df_cam_poses_world, obj_pose_world_coords, rotate_z_axis_180=False):
    obj_poses_in_camera_coords = {}
    camera_poses_in_world_coords = {}

    for i, row in df_cam_poses_world.iterrows():
        trans = np.asarray([[row[1], row[2], row[3], 1.]]).T

        rot3x3 = np.asarray(Rotation.from_quat([row[4], row[5], row[6], row[7]]).as_matrix())

        rot_3x4 = np.vstack((rot3x3, np.zeros((1, rot3x3.shape[0]))))
        projection_matrix = np.hstack((rot_3x4, trans))
        camera_poses_in_world_coords[i] = projection_matrix

        obj_poses_in_camera_coords[i] = np.linalg.inv(projection_matrix) @ obj_pose_world_coords

    return obj_poses_in_camera_coords, camera_poses_in_world_coords

import matplotlib.pyplot as plt
def overlay_input_img_and_rendering(img, render, color=(0, 0, 255), img_intensity=0.6, render_intensity=0.3):
    # Create complete color image
    color_map = np.zeros(img.shape, np.uint8)
    color_map[:] = color

    # Create binary mask from rendering
    mask = (render > 0).astype(bool)

    # Create a colored mask
    colored_mask = mask * color_map

    # Overlay input image with red 2d object projection mask
    overlay_img = cv2.addWeighted(img, img_intensity, colored_mask, render_intensity, 0.0)

    mask = mask.astype(int) * 255

    return overlay_img, mask


def write_on_image(im, text, org=(60,60), font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2, color=(0,0,255)):
    im = cv2.putText(im, text, org, font, fontScale, (255,255,255), thickness+1, cv2.LINE_AA)
    im = cv2.putText(im, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return im


def check_if_images_are_equal(im1, im2):
    diff = cv2.subtract(im1, im2)
    b, g, r = cv2.split(diff)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    else:
        return False


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def create_debug_img(img,
                     model_path,
                     cam_intrinsics,
                     obj_pose_world_cords,
                     cam_pose_world_cords,
                     obj_pose_cam_cords,
                     cam_pose_cam_cords=np.eye(4),
                     color=(0, 0, 255)):
    img_copy = img.copy()
    # render object and project in camera coords into image

    # img = cv2.rotate(img, cv2.ROTATE_180)
    render, depth_cam = project_mesh_to_2d(model_path,
                                           obj_pose_cam_cords,
                                           cam_intrinsics,
                                           cam_pose_cam_cords)

    debug_cam_cords, mask_cam_cords = overlay_input_img_and_rendering(img, render)

    # render object and project camera in world coordinates
    render, depth_world_cords = project_mesh_to_2d(model_path,
                                                   obj_pose_world_cords,
                                                   cam_intrinsics,
                                                   cam_pose_world_cords)

    depth_cam_cords = np.float32(np.zeros_like(img))
    depth_cam_cords[:, :, 0] = depth_cam
    depth_cam_cords[:, :, 1] = depth_cam
    depth_cam_cords[:, :, 2] = depth_cam

    debug_world_cords, mask_world_cords = overlay_input_img_and_rendering(img_copy, render, color)

    return cv2.hconcat([debug_cam_cords, debug_world_cords]), mask_cam_cords, depth_cam_cords


def create_debug_img_test(img,
                          model_path,
                          cam_intrinsics,
                          obj_pose_world_cords, cam_pose_world_cords,
                          obj_pose_cam_cords, cam_pose_cam_cords=np.eye(4),
                          color=(0, 0, 255)):
    img_copy = img.copy()
    cam_pose_world_cords_copy = cam_pose_world_cords.copy()
    # render object and project in camera coords into image

    # render object and project camera in world coordinates
    # img = cv2.rotate(img, cv2.ROTATE_180)
    # cam_pose_world_cords[:3, :3] = cam_pose_world_cords[:3, :3] @ Rotation.from_euler('z', 180,degrees=True).as_matrix()
    render, depth = project_mesh_to_2d(model_path,
                                       obj_pose_world_cords,
                                       cam_intrinsics,
                                       cam_pose_world_cords)

    debug_cam_cords = overlay_input_img_and_rendering(img, render)
    debug_cam_cords = cv2.rotate(debug_cam_cords, cv2.ROTATE_180)

    # render object and project camera in world coordinates
    img_copy = cv2.rotate(img_copy, cv2.ROTATE_180)
    cam_pose_world_cords_copy[:3, :3] = cam_pose_world_cords_copy[:3, :3] @ Rotation.from_euler('z', 180,
                                                                                                degrees=True).as_matrix()
    cam_intrinsics_copy = edict(cam_intrinsics.copy())

    # print(f"delta x-axis:{cam_intrinsics.width/2 - cam_intrinsics.cx}")
    # print(f"delta y-axis:{cam_intrinsics.height/2 - cam_intrinsics.cy}")

    cam_intrinsics_copy.cx += 2 * (cam_intrinsics.width / 2 - cam_intrinsics.cx)
    cam_intrinsics_copy.cy += 2 * (cam_intrinsics.height / 2 - cam_intrinsics.cy)
    render, depth = project_mesh_to_2d(model_path,
                                       obj_pose_world_cords,
                                       cam_intrinsics_copy,
                                       cam_pose_world_cords_copy)

    debug_world_cords = overlay_input_img_and_rendering(img_copy, render, color)

    dublicate_check = False

    if dublicate_check:
        if check_if_images_are_equal(debug_cam_cords, debug_world_cords):
            print("OK")
        else:
            print("FALSE")

    return cv2.hconcat([debug_cam_cords, debug_world_cords])


if __name__ == '__main__':
    # Define paths for source and targets
    # Define paths for source and targets
    # base_path = "/home/philipp/Projects/PhD_projects/Data_Tracebot/Tracebot_Philipp_2022_08_04"
    # base_path = "/home/philipp/Projects/PhD_projects/Data_Tracebot/PHD_V4R"
    base_path = "/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04"
    bop_path = "/home/negar/Documents/Tracebot/Tracebot_Negar_BOP"
    object_names = ["container", "draintray"]
    object_ids = ["1", "2"]
    object_id_dict = {"1": 1, "2": 2}
    store_debug_images = True
    #cam_file_path = f"{base_path}/scenes/001/camera_d435.yaml"
    cam_file = "camera_d435.yaml"
    model_path = f"{base_path}/objects/canister.ply"
    flip_upside_down = False
    mode = "train"                      # train/val

    scenes_to_flip = [] # ['001', '002', '003']

    # create bop tracebot folders
    os.makedirs(bop_path, exist_ok=True)
    os.makedirs(os.path.join(bop_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(bop_path, "models_eval"), exist_ok=True)

    # create model_info for models and models_eval folder bop format
    models_info = dict()
    test_targets = []
    for oi, object_name in enumerate(object_names):
        model_path_base = os.path.join(base_path, f"objects/{object_name}/{object_name}.ply")
        #model_path_bop = os.path.join(bop_path, f"models/obj_{oi + 1:06d}.ply")
        model_path_bop = os.path.join(bop_path, f"models/obj_{object_id_dict[str(oi+1)]:06d}.ply")

        if not os.path.exists(model_path_bop):
            mesh = trimesh.load(model_path_base)
            # save mesh to bop folder
            # not sure if mesh should be stored in mm or m # TODO: I think mine are in m => scale to 1000 as well
            # if object_name != 'draintray': # Because draintray needs to be scaled down by 0.001 => it's already in mm
            mesh = mesh.apply_scale(1000)  # to mm
            mesh.export(model_path_bop)  # TODO: for now, both objects are good. We can scale down the evaluation later to make it more realistic
            #mesh.export(os.path.join(bop_path, f"models_eval/obj_{oi + 1:06d}.ply"))
            mesh.export(os.path.join(bop_path, f"models_eval/obj_{object_id_dict[str(oi+1)]:06d}.ply"))

            # info on dimensions, symmetries, etc
            min_x, min_y, min_z, max_x, max_y, max_z = mesh.bounds.reshape(-1)
            size_x, size_y, size_z = mesh.extents
            samples = trimesh.sample.sample_surface_even(mesh, 10000)[0]
            diameter = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1).max()
            models_info[object_id_dict[str(oi + 1)]] = {
                'diameter': diameter,
                'min_x': min_x, 'min_y': min_y, 'min_z': min_z,
                'max_x': max_x, 'max_y': max_y, 'max_z': max_z,
                'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
            }

            if oi + 1 in [4, 5]:  # 90deg discrete # TODO: Which one is this ?
                symmetries = np.eye(4)[None, ...].repeat(3, axis=0)
                symmetries[:, :3, :3] = [Rotation.from_euler('z', z, degrees=True).as_matrix() for z in [90, 180, 270]]
                models_info[object_id_dict[str(oi + 1)]]['symmetries_discrete'] = [sym.reshape(-1).tolist() for sym in symmetries]
            elif oi + 1 in [5, 6]:  # continuous
                models_info[object_id_dict[str(oi + 1)]]['symmetries_continuous'] = [{'axis': [0, 0, 1], 'offset': [0, 0, 0]}]  # z-axis
            elif oi + 1 in [2]:  # 180deg discrete, y-axis
                symmetries = np.eye(4)[None, ...]
                symmetries[:, :3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
                models_info[object_id_dict[str(oi + 1)]]['symmetries_discrete'] = [sym.reshape(-1).tolist() for sym in symmetries]
            else:  # 180deg discrete, z-axis
                symmetries = np.eye(4)[None, ...]
                symmetries[:, :3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
                models_info[object_id_dict[str(oi + 1)]]['symmetries_discrete'] = [sym.reshape(-1).tolist() for sym in symmetries]

    if not os.path.exists(f"{bop_path}/models/models_info.json"):
        with open(f"{bop_path}/models/models_info.json", 'w') as file:
            json.dump(models_info, file, indent=2)
        with open(f"{bop_path}/models_eval/models_info.json", 'w') as file:
            json.dump(models_info, file, indent=2)


    # for scene in os.scandir(f"{base_path}/scenes"):
    for scene_num in [1, 2, 3]:
        scene_number = f"{scene_num:03d}"
        scene_path = f"{base_path}/scenes/{scene_number}"
        print(f"{scene_number} in progress:")

        if scene_number in scenes_to_flip:
            flip_upside_down = True
        else:
            flip_upside_down = False

        # if scene_number != "003":
        #     continue

        scene_path_bop = os.path.join(bop_path, f"{mode}/{int(scene_number):06d}")
        os.makedirs(scene_path_bop, exist_ok=True)
        os.makedirs(os.path.join(scene_path_bop, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(scene_path_bop, "depth"), exist_ok=True)
        os.makedirs(os.path.join(scene_path_bop, "mask"), exist_ok=True)
        os.makedirs(os.path.join(scene_path_bop, "dbg"), exist_ok=True)

        # open camera poses in world coordinates trans|quaternion repr (3|4)
        with open(f"{scene_path}/groundtruth_handeye.txt", "r") as f:
            df_cam_poses_world = pd.read_table(f, sep=' ',
                                               index_col=0,
                                               header=None,
                                               lineterminator='\n')

        # open object pose in world coordinates for scene as (R|T) (3x4|1x4)
        with open(f"{scene_path}/poses.yaml", "r") as file:
            obj_poses = yaml.safe_load(file)

        scene_objects_name = [obj["id"] for obj in obj_poses]

        with open(f"{base_path}/scenes/{scene_number}/{cam_file}", "r") as file:
            camera_intrinsics = yaml.safe_load(file)

        cam_m = np.array(camera_intrinsics['camera_matrix']).reshape((3, 3))

        cam_intrinsics = edict({"fx": cam_m[0, 0],
                                "fy": cam_m[1, 1],
                                "cx": cam_m[0, 2],
                                "cy": cam_m[1, 2],
                                "width": camera_intrinsics['image_width'],
                                "height": camera_intrinsics['image_height']})

        if flip_upside_down:
            cam_intrinsics.cx += 2 * (cam_intrinsics.width / 2 - cam_intrinsics.cx)
            cam_intrinsics.cy += 2 * (cam_intrinsics.height / 2 - cam_intrinsics.cy)

        with open(os.path.join(bop_path, "camera.json"), 'w') as file:
            json.dump(cam_intrinsics, file, indent=2)

        # transform object from world into camera coordinates
        cam_poses_world_cords = transform_cam_poses_to_matrix_form(df_cam_poses_world)

        obj_poses_cam_cords_all_images = {}
        obj_poses_world_cords = {} # from poses.yaml
        for img_number in cam_poses_world_cords.keys():
            if flip_upside_down:
                cam_poses_world_cords[img_number][:3, :3] = cam_poses_world_cords[img_number][:3, :3] @ \
                                                            Rotation.from_euler('z', 180, degrees=True).as_matrix()

            obj_poses_per_img = {}
            # For each object in the scene

            for obj_pose in obj_poses:
                obj_id = obj_pose['id'] # The objects name
                obj_pose_world_cords = np.array(obj_pose['pose']).reshape((4, 4)) # Coordinate of the single object in world

                if obj_id in obj_poses_world_cords:
                    obj_poses_world_cords[obj_id].append(obj_pose_world_cords) # If we have two instance from one object, store them in a same one
                else:
                    obj_poses_world_cords[obj_id] = [obj_pose_world_cords]

                if obj_id in obj_poses_per_img:
                    obj_poses_per_img[obj_id].append(np.linalg.inv(cam_poses_world_cords[img_number]) @
                                                     obj_pose_world_cords) # If we have two instance from one object, store them in a same one
                else:
                    obj_poses_per_img[obj_id] = [np.linalg.inv(cam_poses_world_cords[img_number]) @
                                                 obj_pose_world_cords]

            obj_poses_cam_cords_all_images[img_number] = obj_poses_per_img

        scene_cameras = dict()
        scene_gts = dict()

        for i, img_file in enumerate(tqdm(os.scandir(f"{scene_path}/rgb"))):

            if img_file.is_file():
                if img_file.path.endswith(".png"):
                    img_number = int(img_file.path.split("/")[-1].split(".")[0])
                    img_name = img_file.path.split("/")[-1]
                else:
                    continue

                img = cv2.imread(img_file.path)
                depth = cv2.imread(os.path.join(scene_path, f"depth/{img_name}"))


                if flip_upside_down:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                    depth = cv2.rotate(depth, cv2.ROTATE_180)

                cv2.imwrite(os.path.join(scene_path_bop, f"rgb/{img_name}"), img)
                cv2.imwrite(os.path.join(scene_path_bop, f"depth/{img_name}"), depth)

                # scene camera extrinsics in world coordinates and intrinsics
                cam_R_floats = [float(v) for v in cam_poses_world_cords[img_number][:3, :3].reshape(-1)]
                cam_t_floats = [float(v) * 1000 for v in cam_poses_world_cords[img_number][:3, 3].reshape(-1)]

                # prepare and store scene camera to bop
                K = np.array([[cam_intrinsics.fx, 0, cam_intrinsics.cx],
                              [0, cam_intrinsics.fy, cam_intrinsics.cy],
                              [0, 0, 1.0]])

                scene_cameras[str(img_number)] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0,
                                                  "cam_R_w2c": cam_R_floats, "cam_t_w2c": cam_t_floats}

                scene_gts[str(img_number)] = []
                instances = dict()

                #for obj_idx, obj_name in enumerate(object_names):
                try:
                    obj_pose_cam_cords = obj_poses_cam_cords_all_images[img_number]
                except:
                    print("Here we are")
                # This is stupid has to be changed later because images are not counted per object type they are
                # counted overall
                counter = 0
                for obj_id, obj_name in zip(object_ids, object_names):
                    # poses_per_obj = obj_pose_cam_cords[obj_id]
                    if obj_name not in scene_objects_name:
                        continue
                    poses_per_obj = obj_pose_cam_cords[obj_name]
                    # model_path = os.path.join(base_path, "objects", f"/{obj_name}/{obj_name}.ply")
                    model_path = os.path.join(base_path, f"objects/{obj_name}/{obj_name}.ply")

                    for obj_quant, pose in enumerate(poses_per_obj):
                        mask = cv2.imread(os.path.join(scene_path, f"masks/003_{obj_name}_{counter:03d}_{img_name}"))
                        #model_path = os.path.join(bop_path, "models", f"obj_{int(obj_id):06d}.ply")
                        if flip_upside_down:
                            mask = cv2.rotate(mask, cv2.ROTATE_180)

                        # cv2.imwrite(os.path.join(scene_path_bop, f"mask/{int(obj_id):03d}_{obj_quant:03d}_{img_name}"),
                        #             mask)
                        cv2.imwrite(os.path.join(scene_path_bop, f"mask/{object_id_dict[obj_id]:03d}_{obj_quant:03d}_{img_name}"),
                                    mask)
                        counter += 1

                        R_floats = [float(v) for v in pose[:3, :3].reshape(-1)]
                        t_floats = [float(v) * 1000 for v in pose[:3, 3].reshape(-1)]  # mm
                        # scene_gts[str(img_number)] += [
                        #     {"cam_R_m2c": R_floats, "cam_t_m2c": t_floats, "obj_id": obj_id}]
                        scene_gts[str(img_number)] += [
                            {"cam_R_m2c": R_floats, "cam_t_m2c": t_floats, "obj_id": object_id_dict[obj_id]}]

                        # creating debug images with object rendered in world coordinates and camera coordinates
                        if store_debug_images:
                            if i % 10 == 0:
                                # try:
                                dbg_img, mask_proj, depth_proj = create_debug_img(img,
                                                                                  model_path,
                                                                                  cam_intrinsics,
                                                                                  obj_poses_world_cords[obj_name][obj_quant], # TODO: I changed here
                                                                                  cam_poses_world_cords[img_number],
                                                                                  pose)



                                depth_proj *= 1000
                                depth_proj = depth_proj.astype(np.uint8)
                                vis_mask = bop_vis.estimate_visib_mask_est(depth, depth_proj, mask, delta=1)
                                vis_mask = vis_mask.astype(np.uint8) * 255

                                # alpha Contrast control (1.0-3.0), beta Brightness control (0-100)
                                #depth_adjusted = cv2.convertScaleAbs(depth, alpha=1.0, beta=15)
                                depth_adjusted = depth * 100
                                depth_adjusted = depth_adjusted.astype(np.uint8)
                                depth_proj = cv2.convertScaleAbs(depth_proj, alpha=1.0, beta=15)

                                dbg_img = write_on_image(dbg_img, "cam cords", org=(60,60), fontScale=2, thickness=2, color=(0,0,255))
                                dbg_img = write_on_image(dbg_img, "world cords", org=(img.shape[1]+60,60))
                                mask = write_on_image(mask, "mask_gt")
                                depth_adjusted = write_on_image(depth_adjusted, "depth_gt")
                                depth_proj = write_on_image(depth_proj, "rendered depth")
                                vis_mask = write_on_image(vis_mask, "visibility mask")

                                dbg_img_with_depth = cv2.vconcat([cv2.hconcat([dbg_img, mask]),
                                                                  cv2.hconcat([depth_adjusted, depth_proj, vis_mask])])

                                cv2.imwrite(f"{scene_path_bop}/dbg/{obj_name}_{obj_quant:03d}_{img_name}", dbg_img_with_depth)

                        if obj_id not in instances:
                            instances[obj_id] = 0
                        instances[obj_id] += 1

                    for obj_id, inst_count in instances.items():
                        test_targets.append({"im_id": img_number, "inst_count": inst_count, "obj_id": obj_id,
                                             "scene_id": scene_number})

            # write meta files
            with open(f"{scene_path_bop}/scene_gt.json", 'w') as file:
                json.dump(scene_gts, file, indent=2)
            with open(f"{scene_path_bop}/scene_camera.json", 'w') as file:
                json.dump(scene_cameras, file, indent=2)


    with open(f"{bop_path}/{mode}_targets.json", 'w') as file:
        json.dump(test_targets, file, indent=2)