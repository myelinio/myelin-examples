import tensorflow as tf
import sys

import argparse
import ocr_dataset


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    parser.prog = 'aocr'
    subparsers = parser.add_subparsers()

    # Global arguments
    parser_base = argparse.ArgumentParser(add_help=False)
    parser_base.add_argument('--log-path', dest="log_path",
                             metavar=defaults.LOG_PATH,
                             type=str, default=defaults.LOG_PATH,
                             help=('log file path (default: %s)'
                                   % (defaults.LOG_PATH)))

    # Dataset generation
    parser_dataset = subparsers.add_parser('dataset', parents=[parser_base],
                                           help='create a dataset in the TFRecords format')
    parser_dataset.set_defaults(phase='dataset')
    parser_dataset.add_argument('annotations_path', metavar='annotations',
                                type=str,
                                help=('path to the annotation file'))
    parser_dataset.add_argument('output_path', nargs='?', metavar='output',
                                type=str, default=defaults.NEW_DATASET_PATH,
                                help=('output path (default: %s)'
                                      % defaults.NEW_DATASET_PATH))
    parser_dataset.add_argument('--log-step', dest='log_step',
                                type=int, default=defaults.LOG_STEP,
                                metavar=defaults.LOG_STEP,
                                help=('print log messages every N steps (default: %s)'
                                      % defaults.LOG_STEP))
    parser_dataset.add_argument('--no-force-uppercase', dest='force_uppercase',
                                action='store_false', default=defaults.FORCE_UPPERCASE,
                                help='do not force uppercase on label values')
    parser_dataset.add_argument('--save-filename', dest='save_filename',
                                action='store_true', default=defaults.SAVE_FILENAME,
                                help='save filename as a field in the dataset')

    parameters = parser.parse_args(args)
    return parameters


class Config(object):
    """
    Default config (see __main__.py or README for documentation).
    """

    GPU_ID = 0
    VISUALIZE = False

    # I/O
    NEW_DATASET_PATH = './dataset.tfrecords'
    DATA_PATH = './data.tfrecords'
    MODEL_DIR = './checkpoints'
    LOG_PATH = 'aocr.log'
    OUTPUT_DIR = './results'
    STEPS_PER_CHECKPOINT = 100
    EXPORT_FORMAT = 'savedmodel'
    EXPORT_PATH = 'exported'
    FORCE_UPPERCASE = True
    SAVE_FILENAME = False
    FULL_ASCII = False

    MAX_WIDTH = 100 * 4
    MAX_HEIGHT = 32 * 4
    MAX_PREDICTION = 40

    USE_DISTANCE = True

    # Dataset generation
    LOG_STEP = 500


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ocr_dataset.generate(
            parameters.annotations_path,
            parameters.output_path,
            parameters.log_step,
            parameters.force_uppercase,
            parameters.save_filename
        )


if __name__ == "__main__":
    main()
