import argparse

import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)
    return sub_im


def predict(east_detect, img_path, pixel_threshold, quiet=False, save=True):
    img = image.load_img(img_path)
    return predict_img(east_detect, img, pixel_threshold, img_path, quiet, save)


def predict_np(east_detect, img_np, pixel_threshold):
    img = cv2.imdecode(img_np, 1)
    img_pil = Image.fromarray(img)
    return predict_img(east_detect, img_pil, pixel_threshold, None)


def predict_img(east_detect, img, pixel_threshold, save_path, quiet=False, save=True):
    im = img
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    im_array = image.img_to_array(im.convert('RGB'))
    d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / im.width
    scale_ratio_h = d_height / im.height
    im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    quad_im = im.copy()
    draw = ImageDraw.Draw(im)
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = (j + 0.5) * cfg.pixel_size
        py = (i + 0.5) * cfg.pixel_size
        line_width, line_color = 1, 'red'
        if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            if y[i, j, 2] < cfg.trunc_threshold:
                line_width, line_color = 2, 'yellow'
            elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                line_width, line_color = 2, 'green'
        draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                  width=line_width, fill=line_color)
    if save:
        im.save(save_path + '_act.jpg')
    quad_draw = ImageDraw.Draw(quad_im)
    txt_items = []
    sub_imgs = []
    for score, geo, s in zip(quad_scores, quad_after_nms,
                             range(len(quad_scores))):
        if np.amin(score) > 0:
            quad_draw.line([tuple(geo[0]),
                            tuple(geo[1]),
                            tuple(geo[2]),
                            tuple(geo[3]),
                            tuple(geo[0])], width=2, fill='red')
            if cfg.predict_cut_text_line:
                sub_im = cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                       save_path[:-4], s)
                sub_imgs.append(sub_im)
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if save:
        quad_im.save(save_path[:-4] + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(save_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)

    return {"quad_im": quad_im, "txt_items": txt_items, "sub_imgs": sub_imgs}


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/f8.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    parser.add_argument('--model_path', default=cfg.model_path, type=str)
    parser.add_argument('--data_dir', default=cfg.data_dir, type=str)
    parser.add_argument('--train_task_id', default=cfg.train_task_id, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)

    model_weights_path = args.model_path + 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                         % args.model_path
    saved_model_file_path = args.model_path + 'saved_model/east_model_%s.h5' % args.train_task_id
    saved_model_weights_file_path = args.model_path + 'saved_model/east_model_weights_%s.h5' \
                                    % args.train_task_id

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)
