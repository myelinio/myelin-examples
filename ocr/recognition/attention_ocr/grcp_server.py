"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import datasets
import data_provider
import os
import tempfile

import ocr_common_pb2
import ocr_recognition_pb2
import ocr_recognition_pb2_grpc

FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_string('image_path_pattern', '', 'A file pattern with a placeholder for the image index.')
flags.DEFINE_integer('port', 8770,'port')
flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
FLAGS.dataset_name = 'floorplan'

def get_dataset_image_size(dataset_name):
    # Ideally this info should be exposed through the dataset interface itself.
    # But currently it is not available by other means.
    ds_module = getattr(datasets, dataset_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
    num_views = ds_module.DEFAULT_CONFIG['num_of_views']
    return width, height, num_views


batch_size = 1
im_w, im_h, num_of_views = get_dataset_image_size(FLAGS.dataset_name)
im_size = (im_w, im_h)


def create_model(batch_size):
    width, height = im_size
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)
    raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
    images = tf.map_fn(data_provider.preprocess_image, raw_images,
                       dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints









###################################################################
from PIL import Image, ImageEnhance
import random
from scipy import ndimage


def concat_n_images(images):
    w = sum(i.size[0] for i in images)
    mh = max(i.size[1] for i in images)

    result = Image.new("RGBA", (w, mh))

    x = 0
    for i in images:
        result.paste(i, (x, 0))
        x += i.size[0]

    return result


def transform_simple(img):
    denoised = Image.fromarray(ndimage.gaussian_filter(img, 2))
    denoised = change_contrast(denoised, 100)
    denoised2 = Image.fromarray(ndimage.median_filter(img, 2))

    return [denoised,
            denoised2,
            rotate(denoised, random.randint(1, 5)),
            rotate(denoised, random.randint(-5, -1))
            ]


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


def convert_to_aocr(original_arr):
    original_im = Image.fromarray(original_arr)

    original_im = original_im.convert('RGB')
    origin_width, _ = original_im.size
    aliases = [Image.ANTIALIAS, Image.NEAREST, Image.BILINEAR, Image.BICUBIC]

    single_im_size = (im_size[0] / num_of_views, im_size[1])
    img_resized = original_im.resize(single_im_size, aliases[random.randint(0, 3)])
    imgs_trans = transform_simple(img_resized)
    imgs_trans = [x.resize(single_im_size, aliases[random.randint(0, 3)]) for x in imgs_trans]
    im = concat_n_images(random.sample(imgs_trans, 4))
    im = im.convert('RGB')

    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    f.close()
    im.save(f.name, format="png")
    im.close()

    im_png = np.array(Image.open(f.name))
    os.remove(f.name)
    return im_png

###################################################################
### the webserver
import argparse


import time
import grpc
from concurrent import futures

def extract_img(pb2_img):
    message_image = np.ndarray.tobytes(np.frombuffer(pb2_img.raw, dtype=np.uint8))
    np_img = np.frombuffer(message_image, dtype=np.uint8)
    shape = pb2_img.shape
    img_shape = (shape['h'], shape['w'], shape['d'])
    return np.reshape(np_img, img_shape)


def build_proto_img(img):
    message_image = np.ndarray.tobytes(img)
    h, w, d = img.shape
    return ocr_common_pb2.Image(raw=message_image, shape={'w': w, 'h': h, 'd': d})


class Recogniser:
    def __init__(self, sess, endpoints, images_placeholder):
        self.sess = sess
        self.endpoints = endpoints
        self.images_placeholder = images_placeholder

    def RecogniseText(self, request, context):
        request_img = extract_img(request.image)
        img = convert_to_aocr(request_img)

        predictions = self.sess.run(self.endpoints.predicted_text,
                               feed_dict={self.images_placeholder: [img]})

        label = predictions.tolist()[0]

        return ocr_recognition_pb2.RecognitionReply(label=label)


def main():
    checkpoint_path = os.getenv("TF_MODEL_DIR", '/root/model/')
    print("checkpoint_path: %s" % checkpoint_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8770, type=int)
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_path)


    images_placeholder, endpoints = create_model(batch_size)
    session_creator = monitored_session.ChiefSessionCreator(
        checkpoint_filename_with_path=checkpoint_path)
    sess = monitored_session.MonitoredSession(
        session_creator=session_creator)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    ocr_recognition_pb2_grpc.add_OcrRecognitionServiceServicer_to_server(
        Recogniser(sess, endpoints, images_placeholder), server)

    print('Starting server. Listening on port %s.' % args.port)
    server.add_insecure_port('[::]:%s' % args.port)
    server.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()