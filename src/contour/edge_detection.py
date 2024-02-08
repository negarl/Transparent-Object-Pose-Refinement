import os
import src.config as config
from PIL import Image, ImageFilter
import numpy as np
import cv2


def find_the_best_outside_edge():
    dataset_path = os.path.join(config.PATH_DATASET_TRACEBOT)
    scene_name = "002"
    scene_path = os.path.join(dataset_path, f"scenes/{scene_name}")
    im_id = 1

    image_rgb = Image.open(os.path.join(scene_path, f"rgb/{im_id:06d}.png"))
    image_mask_1 = Image.open(os.path.join(scene_path, f"masks/003_container_{0:03d}_{im_id:06d}.png"))
    image_mask_2 = Image.open(os.path.join(scene_path, f"masks/003_container_{1:03d}_{im_id:06d}.png"))

    # Converting the image to grayscale, as edge detection
    # requires input image to be of mode = Grayscale (L)
    image_rgb = image_rgb.convert("L")

    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    image_rgb = image_rgb.filter(ImageFilter.FIND_EDGES)

    # Saving the Image Under the name Edge_Sample3_102.png
    # image_rgb.save(r"Edge_Sample3_006 .png")

    image_rgb = np.asarray(image_rgb)
    image_mask_1 = np.asarray(image_mask_1)
    image_mask_2 = np.asarray(image_mask_2)
    x = np.where(image_mask_1 == 0, 0, image_rgb)
    y = np.where(image_mask_2 == 0 , 0 , image_rgb)
    z = x + y
    im = Image.fromarray(np.uint8((z)))
    im.save(r"only_objects_edge.png")

    IM = Image.open("only_objects_edge.png")
    sharp_img = IM.filter(ImageFilter.SHARPEN)
    smooth_img = sharp_img.filter(ImageFilter.SMOOTH)
    smooth_img.show()

    open_cv_image = np.array(smooth_img)
    mask = np.zeros(open_cv_image.shape, dtype=np.uint8) * 255
    # gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(open_cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=2)

    cv2.imwrite("out_edges.png", mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()


def single_object_edge(scene_path, im_id, obj_num):

    image_rgb = Image.open(os.path.join(scene_path, f"rgb/{im_id:06d}.png"))
    image_mask_1 = Image.open(os.path.join(scene_path, f"masks/003_container_{obj_num:03d}_{im_id:06d}.png"))

    # Converting the image to grayscale, as edge detection
    # requires input image to be of mode = Grayscale (L)
    image_rgb = image_rgb.convert("L")

    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    image_rgb = image_rgb.filter(ImageFilter.FIND_EDGES)

    # Saving the Image Under the name Edge_Sample3_102.png
    image_rgb.save(rf"Edge_Sample3_006_{obj_num}.png")

    image_rgb = np.asarray(image_rgb)
    image_mask_1 = np.asarray(image_mask_1)
    x = np.where(image_mask_1 == 0, 0, image_rgb)
    im = Image.fromarray(np.uint8((x)))
    im.save(rf"only_objects_edge_{obj_num}.png")

    # Sharping and Smoothing the image for getting a clearer edges.
    IM = Image.open(f"only_objects_edge_{obj_num}.png")
    sharp_img = IM.filter(ImageFilter.SHARPEN)
    smooth_img = sharp_img.filter(ImageFilter.SMOOTH)
    edge_enhance = smooth_img.filter(ImageFilter.EDGE_ENHANCE)
    edge_enhance.show()

    open_cv_image = np.array(smooth_img)
    mask = np.zeros(open_cv_image.shape, dtype=np.uint8) * 255
    # gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(open_cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=1)

    cv2.imwrite(f"out_edges_{obj_num}.png", mask)


def inner_edge_single_image(scene_path, im_id, obj_num):

    img = cv2.imread('only_objects_edge_1.png')
    mask = cv2.imread('out_edges_1.png')
    image_mask_1 = cv2.imread(os.path.join(scene_path, f"masks/003_container_{obj_num:03d}_{im_id:06d}.png"))
    result = cv2.bitwise_xor(img, mask, img,  mask=image_mask_1)
    cv2.imshow('mask', result)
    cv2.waitKey()


if __name__ == '__main__':
    dataset_path = os.path.join(config.PATH_DATASET_TRACEBOT)
    scene_name = "002"
    scene_path = os.path.join(dataset_path, f"scenes/{scene_name}")
    im_id = 1
    single_object_edge(scene_path, im_id, 1)

    inner_edge_single_image(scene_path, im_id, 1)

    print("THE END")