import argparse

import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm

import cfg
from label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def transform(img):
    width_r, height_r = img.size
    quad_shift = height_r // 50
    return [img,
            # rotate(img, random.randint(1, 4)),
            # rotate(img, random.randint(-4, -1)),
            noisy(img, 'gauss'),
            noisy(img, 'poisson'),
            noisy(img, 's&p'),
            # noisy(img, 'speckle'),
            quad_transform(img, quad_shift, -quad_shift),
            quad_transform(img, -quad_shift, quad_shift),
            change_contrast(img, random.randint(50, 100)),
            change_brightness(img, random.uniform(0.5, 1.9))
            ]

from PIL import Image, ImageEnhance


def change_brightness(img, level):
    contrast = ImageEnhance.Contrast(img)
    return contrast.enhance(level)


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


def rotate(img, angle):
    # original image
    # converted to have an alpha layer
    im2 = img.convert('RGBA')
    # rotated image
    rot = im2.rotate(angle, expand=1)
    # a white image same size as rotated image
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)
    # save your work (converting back to mode='1' or whatever..)
    return out.convert(img.mode)


def quad_transform(im, shift_left, shift_right):
    width, height = im.size

    im2 = im.convert('RGBA')
    im2 = im2.transform((im.size), Image.QUAD, (
        0, 0, 0, height + shift_left, width + shift_left, height + shift_right, width + shift_right, 0))
    return fill_white(im, im2)


def mesh_transform(im, shift_left, shift_right):
    im2 = im.convert('RGBA')
    (w, h) = im.size
    im2 = im2.transform((im.size), Image.MESH,
                        [(
                            (0, 0, w, h),  # box
                            (0, 0, 0, h + shift_right, w + shift_right, h + shift_right, w + shift_right, 0))
                        ],  # ul -> ccw around quad
                        Image.BILINEAR)

    return fill_white(im, im2)


def fill_white(im, im2):
    # a white image same size as rotated image
    fff = Image.new('RGBA', im2.size, (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(im2, fff, im2)
    # save your work (converting back to mode='1' or whatever..)
    return out.convert(im.mode)


def noisy(pil_image, noise_typ):
    image = np.array(pil_image)
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return pil_from_np(noisy)
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return pil_from_np(out)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return pil_from_np(noisy)
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return pil_from_np(noisy)


def pil_from_np(np_array):
    return Image.fromarray(np_array.astype('uint8'), 'RGB')


def preprocess(data_dir):
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):

        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')

            imgs_trans = transform(im)
            txt_fname = o_img_fname[:-4] + '.txt'
            init_o_img_fname = o_img_fname
            for im_idx, im in enumerate(imgs_trans):
                show_gt_im = im.copy()
                # draw on the img
                draw = ImageDraw.Draw(show_gt_im)
                o_img_fname = "%s_%s.%s" % (init_o_img_fname[:-4], str(im_idx), o_img_fname[-3:])
                with open(os.path.join(origin_txt_dir,
                                       txt_fname), 'r') as f:
                    anno_list = f.readlines()
                xy_list_array = np.zeros((len(anno_list), 4, 2))
                for anno, i in zip(anno_list, range(len(anno_list))):
                    anno_colums = anno.strip().split(',')
                    anno_array = np.array(anno_colums)
                    xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                    xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                    xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                    xy_list = reorder_vertexes(xy_list)
                    xy_list_array[i] = xy_list
                    _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                    shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                    if draw_gt_quad:
                        draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                                   tuple(xy_list[2]), tuple(xy_list[3]),
                                   tuple(xy_list[0])
                                   ],
                                  width=2, fill='green')
                        draw.line([tuple(shrink_xy_list[0]),
                                   tuple(shrink_xy_list[1]),
                                   tuple(shrink_xy_list[2]),
                                   tuple(shrink_xy_list[3]),
                                   tuple(shrink_xy_list[0])
                                   ],
                                  width=2, fill='blue')
                        vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                              [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                        for q_th in range(2):
                            draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                       tuple(shrink_1[vs[long_edge][q_th][1]]),
                                       tuple(shrink_1[vs[long_edge][q_th][2]]),
                                       tuple(xy_list[vs[long_edge][q_th][3]]),
                                       tuple(xy_list[vs[long_edge][q_th][4]])],
                                      width=3, fill='yellow')
                if cfg.gen_origin_img:
                    im.save(os.path.join(train_image_dir, o_img_fname))
                np.save(os.path.join(
                    train_label_dir,
                    o_img_fname[:-4] + '.npy'),
                    xy_list_array)
                if draw_gt_quad:
                    show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
                train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                         d_wight,
                                                         d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=cfg.data_dir, type=str)
    args = parser.parse_args()
    preprocess(args.data_dir)
