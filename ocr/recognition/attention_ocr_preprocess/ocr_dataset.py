from __future__ import absolute_import

import logging
import traceback

import tensorflow as tf

from six import b

from PIL import Image
from io import BytesIO
import random
from random import randrange
import shutil


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64=tf.train.Int64(value=value))


def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):
    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    longest_label = ''
    idx = 0

    max_width = 0
    max_height = 0
    im_cnt = 0
    charset_file = './charset_size=96.txt'
    charset = read_charset(charset_file)
    max_sequence_length = 30
    NULL_CODE = 95
    # if os.path.isdir(output_path):
    #     shutil.rmtree(output_path)

    # os.makedirs(output_path)

    writer = None
    batch = 10000
    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):

            # if not hasNumbers(label):
            #     r = random.random()
            #     if r > 0.1:
            #         continue

            file_name = "%s-%s" % (output_path, idx // batch)

            if idx % batch == 0:
                if writer is not None:
                    writer.close()
                if os.path.isfile(file_name):
                    logging.warning('Skipping file %s' % file_name)
                    writer = None
                else:
                    writer = tf.python_io.TFRecordWriter(file_name)

            if writer is None:
                continue

            line = line.rstrip('\n')
            try:
                (img_path, label) = line.split(None, 1)


                original_im = Image.open(img_path)
                original_im = original_im.convert('RGB')
                origin_width, _ = original_im.size
                aliases = [Image.ANTIALIAS, Image.NEAREST, Image.BILINEAR, Image.BICUBIC]
                img_resized = original_im.resize((150, 50), aliases[random.randint(0, 3)])
                imgs_trans = transform(img_resized)
                imgs_trans = [x.resize((150, 50), aliases[random.randint(0, 3)]) for x in imgs_trans]
                imgs = [concat_n_images(random.sample(imgs_trans, 4))]

                for im in imgs:
                    img_io = BytesIO()
                    im = im.convert('RGB')
                    im.save(img_io, format="png")

                    # im.save('text_detection/tf/attention_ocr/python/testdata/floorplan/validation_%d.png' % im_cnt, format="png")
                    # if im_cnt > 100:
                    #     break

                    im.close()
                    img = img_io.getvalue()

                    label = label.upper()

                    if len(label) > len(longest_label):
                        longest_label = label

                    encoded = [encode_char(x, charset) for x in label[:max_sequence_length]]
                    encoded_padded = encoded + [NULL_CODE] * (max_sequence_length - len(encoded))

                    assert(len(encoded_padded) == max_sequence_length)
                    feature = {}
                    feature['image/encoded'] = _bytes_feature(img)
                    feature['image/format'] = _bytes_feature(b('PNG'))
                    feature['image/width'] = _int64_list_feature(50)
                    feature['image/height'] = _int64_list_feature(600)
                    feature['image/orig_width'] = _int64_list_feature(origin_width)
                    feature['image/class'] = _int64_list_feature_list(encoded_padded)
                    feature['image/unpadded_class'] = _int64_list_feature_list(encoded)
                    feature['image/text'] = _bytes_feature(b(label))

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    writer.write(example.SerializeToString())
                    im_cnt = im_cnt + 1

            except Exception as e:
                traceback.print_exc()
                logging.error('missing filename or label, ignoring line %i: %s, error: %s', idx + 1, line, str(e))
                continue


            # if im_cnt > 100:
            #     break

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx + 1)
    if writer is not None:
        writer.close()
    if idx:
        logging.info('Dataset is ready: %i pairs.', idx + 1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    logging.info('Max width: %s, height: %s', max_width, max_height)


def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def concat_n_images(images):
    w = sum(i.size[0] for i in images)
    mh = max(i.size[1] for i in images)

    result = Image.new("RGBA", (w, mh))

    x = 0
    for i in images:
        result.paste(i, (x, 0))
        x += i.size[0]

    return result

from scipy import ndimage

def transform_simple(img):
    denoised = Image.fromarray(ndimage.gaussian_filter(img, 2))
    denoised = change_contrast(denoised, 100)
    denoised2 = Image.fromarray(ndimage.median_filter(img, 2))

    return [denoised,
            denoised2,
            rotate(denoised, random.randint(1, 5)),
            rotate(denoised, random.randint(-5, -1))
            ]


def transform(img):
    width_r, height_r = img.size
    quad_shift = height_r // 2
    return [img,
            rotate(img, random.randint(1, 10)),
            rotate(img, random.randint(-10, -1)),
            noisy(img, 'gauss'),
            noisy(img, 'poisson'),
            # noisy(img, 's&p'),
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


import numpy as np
import os
# import cv2


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


import re


def read_charset(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the FLOORPLAN dataset.

    Args:
      filename: a path to the charset file.
      null_character: a unicode character used to replace '<null>' character. the
        default value is a light shade block.

    Returns:
      a dictionary with keys equal to character codes and values - unicode
      characters.
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset

def encode_char(x, charset):
    for i, c in charset.items():
        if x == c:
            return i
    raise Exception('%s not found' % x)


