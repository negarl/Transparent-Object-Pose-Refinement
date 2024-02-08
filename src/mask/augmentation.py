import math

import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def knn(mask_points, nonmask_points, seed_point, k, prob_select_pixel):

    distance_list_mask_points = np.linalg.norm(mask_points - seed_point, axis=1)
    mask_sort_dis_indx = np.argsort(np.asarray(distance_list_mask_points))
    mask_sorted_points = mask_points[mask_sort_dis_indx]

    distance_list_nonmask_points = np.linalg.norm(nonmask_points - seed_point, axis=1)
    nonmask_sort_dis_indx = np.argsort(np.asarray(distance_list_nonmask_points))
    nonmask_sorted_points = nonmask_points[nonmask_sort_dis_indx]

    knn_point_list_remove = mask_sorted_points[:int(k * prob_select_pixel)]
    knn_point_list_add = []

    if 1.0 - prob_select_pixel != 0:
        knn_point_list_add = nonmask_sorted_points[:int(k * (1.0 - prob_select_pixel))]

    return np.asarray(knn_point_list_add), np.asarray(knn_point_list_remove)


def single_mask_augmentation(mask_image, variant_name="knn", perc_of_k=0.5, prob_select_pixel=1.0):
    """

    :param mask_image:
    :param variant_name:
    :param perc_of_k:
    :param prob_select_pixel:
    :return:
    """

    num_mask_pixels = np.sum(mask_image) / 255

    mask_points = np.transpose(np.where(mask_image > 0))
    nonmask_points = np.transpose(np.where(mask_image == 0))

    np.random.seed(0)
    random_point_idx = np.random.randint(len(mask_points), size=1)[0]

    new_mask = np.zeros(mask_image.shape)

    if variant_name == "knn":
        knn_point_list_add, knn_point_list_remove = knn(mask_points, nonmask_points, [mask_points[random_point_idx][0], mask_points[random_point_idx][1]], perc_of_k * num_mask_pixels, prob_select_pixel)

        for i in range(len(knn_point_list_add)):
            new_mask[knn_point_list_add[i][0]][knn_point_list_add[i][1]] = 255

        for i in range(len(knn_point_list_remove)):
            new_mask[knn_point_list_remove[i][0]][knn_point_list_remove[i][1]] = 255

    elif variant_name == "square":  # Occluder
        new_mask = np.copy(mask_image)
        s = math.sqrt(perc_of_k * num_mask_pixels) # Square side size
        x = mask_points[random_point_idx][0]
        y = mask_points[random_point_idx][1]
        new_mask[ x-int(s/2): x+int(s/2), y-int(s/2):y+int(s/2)] = 0

    # cv2.imshow("sf", new_mask)
    # cv2.waitKey(0)
    return new_mask


def mask_augmentation(scene_path):
    for scene_number in np.sort(os.listdir(os.path.join(scene_path))):

        print(scene_number, "____________________________")

        new_mask_path = os.path.join("/home/negar/Documents/Tracebot/Files/diffrend/src/pose/result/new_masks", f"{scene_number}/square_05_08/")
        # new_mask_path = os.path.join(scene_path, f"{scene_number}/new_mask/")

        if not os.path.exists(new_mask_path):
            os.makedirs(new_mask_path)

        for image_name in np.sort(os.listdir(os.path.join(scene_path, f"{scene_number}/masks/"))):
            reference_mask = imageio.imread(os.path.join(scene_path, f"{scene_number}/masks/{image_name}"))
            new_ref_mask = single_mask_augmentation(reference_mask, "square", 0.5, 0.8) #change 0.8 to 1
            cv2.imwrite(os.path.join(new_mask_path, f"{image_name}"), new_ref_mask)

    return


if __name__ == '__main__':
    scene_path = "/home/negar/Documents/Tracebot/Files/canister_scenes/"
    mask_augmentation(scene_path)

    print("the end")
