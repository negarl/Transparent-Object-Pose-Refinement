import os

import numpy as np
import torch
import pandas as pd
import json


def make_result_yaml(result_cv_path, save_path):
    result_df_1 = pd.read_csv(result_cv_path)
    result_df = result_df_1.sort_values(by=['scene_id', 'im_id', 'obj_id'], axis=0,
                                        ascending=[True, True, True],
                                        inplace=False,
                                        kind='quicksort', na_position='first',
                                        ignore_index=False, key=None)
    scene_id_list = result_df['scene_id'].unique()
    for scene_id in scene_id_list:
        scene_dict = {}
        scene_df = result_df[result_df['scene_id'] == scene_id]
        img_id_list = scene_df['im_id'].unique()
        for im_id in img_id_list:
            img_obj_info_list = []
            img_df = scene_df[scene_df['im_id'] == im_id]
            for obj_id in img_df['obj_id'].unique():
                object_dict = {}
                object_dict.setdefault('R')
                object_dict['R'] = np.asarray(img_df[img_df['obj_id'] == obj_id]['R'].str.split().to_numpy()[0],
                                              dtype=float).tolist()
                object_dict.setdefault('t')
                object_dict['t'] = np.asarray(img_df[img_df['obj_id'] == obj_id]['t'].str.split().to_numpy()[0],
                                              dtype=float).tolist()
                object_dict.setdefault('obj_id')
                object_dict['obj_id'] = int(obj_id)
                img_obj_info_list.append(object_dict)

            scene_dict[f"{im_id}"] = img_obj_info_list

        with open(os.path.join(save_path,
                               f"scene_{scene_id}_result_pose_estimation.yaml"),
                  'w') as fou:
            json.dump(scene_dict, fou)
        print("the end of the scene")


def convert_results_bop_to_yaml(result_cv_path, save_path):
    result_df_1 = pd.read_csv(result_cv_path)
    result_df = result_df_1.sort_values(by=['scene_id', 'im_id', 'obj_id'], axis=0,
                                        ascending=[True, True, True],
                                        inplace=False,
                                        kind='quicksort', na_position='first',
                                        ignore_index=False, key=None)
    scene_id_list = result_df['scene_id'].unique()
    for scene_id in scene_id_list:
        scene_dict = {}
        scene_df = result_df[result_df['scene_id'] == scene_id]
        img_id_list = scene_df['im_id'].unique()
        for im_id in img_id_list:
            img_obj_info_list = []
            img_df = scene_df[scene_df['im_id'] == im_id]
            for obj_id in img_df['obj_id'].unique():
                object_dict = {}
                object_dict.setdefault('R')
                object_dict['R'] = np.asarray(img_df[img_df['obj_id'] == obj_id]['R'].str.split().to_numpy()[0],
                                              dtype=float).tolist()
                object_dict.setdefault('t')
                object_dict['t'] = np.asarray(img_df[img_df['obj_id'] == obj_id]['t'].str.split().to_numpy()[0],
                                              dtype=float).tolist()
                object_dict.setdefault('obj_id')
                object_dict['obj_id'] = int(obj_id)
                img_obj_info_list.append(object_dict)

            scene_dict[f"{im_id}"] = img_obj_info_list

        with open(os.path.join(save_path,
                               f"scene_{scene_id}_result_pose_estimation.yaml"),
                  'w') as fou:
            json.dump(scene_dict, fou)
        print("the end of the scene")

    print("jio")


if __name__ == '__main__':
    # file_without_refine = '/home/negar/Documents/Tracebot/Files/other_methods_results/pix2pose_results/pix2pose-resnet-iccv19_tless-test_without_refinement.csv'
    # file_with_refine = '/home/negar/Documents/Tracebot/Files/other_methods_results/pix2pose_results/pix2pose-resnet-wicp-iccv19_tless-test_icp_refienement.csv'
    # save_path = '/home/negar/Documents/Tracebot/Files/other_methods_results/pix2pose_results'
    # file_with_refine = '/home/negar/Documents/Tracebot/Files/other_methods_results/MegaPose_results/cnos-fastsammegapose-multihyp-teaserpp_tless-test_with_refienement.csv'
    file_without_refine = '/home/negar/Documents/Tracebot/Files/other_methods_results/MegaPose_results/megapose-cnos-fastsamcoarsebest_tless-test_without_refienement.csv'
    save_path = '/home/negar/Documents/Tracebot/Files/other_methods_results/pix2pose_results'
    convert_results_bop_to_yaml(file_without_refine, save_path)
