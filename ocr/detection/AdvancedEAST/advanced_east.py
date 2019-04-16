import argparse
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import cfg
from network import East
from losses import quad_loss
from data_generator import gen

import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=cfg.lr, type=float)
    parser.add_argument('--decay', default=cfg.decay, type=float)
    parser.add_argument('--validation_split_ratio', default=cfg.validation_split_ratio, type=float)
    parser.add_argument('--epoch_num', default=cfg.epoch_num, type=int)
    parser.add_argument('--initial_epoch', default=cfg.initial_epoch, type=int)
    parser.add_argument('--patience', default=cfg.patience, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--model_path', default=cfg.model_path, type=str)
    parser.add_argument('--data_dir', default=cfg.data_dir, type=str)
    # parser.add_argument('--model_weights_path', default=cfg.model_weights_path, type=str)
    # parser.add_argument('--saved_model_file_path', default=cfg.saved_model_file_path, type=str)
    parser.add_argument('--train_task_id', default=cfg.train_task_id, type=str)
    return parser.parse_args()


# --saved_model_file_path=./saved_model/  --model_weights_path=./model/ --data_dir=./icpr
if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.model_path + 'model'):
        os.mkdir(args.model_path + 'model')
    if not os.path.exists(args.model_path + 'saved_model'):
        os.mkdir(args.model_path + 'saved_model')

    logger.info("Config: %s" % args)

    model_weights_path = args.model_path + 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                         % args.model_path
    saved_model_file_path = args.model_path + 'saved_model/east_model_%s.h5' % args.train_task_id
    saved_model_weights_file_path = args.model_path + 'saved_model/east_model_weights_%s.h5' \
                                    % args.train_task_id

    logger.info("model_weights_path: %s" % model_weights_path)
    logger.info("saved_model_file_path: %s" % saved_model_file_path)
    logger.info("saved_model_weights_file_path: %s" % saved_model_weights_file_path)

    total_img = cfg.total_img(args.data_dir)
    logger.info("total_img: %s" % total_img)

    east = East()
    east_network = east.east_network()
    east_network.summary()
    east_network.compile(loss=quad_loss, optimizer=Adam(lr=args.lr,
                                                        # clipvalue=cfg.clipvalue,
                                                        decay=args.decay))
    if cfg.load_weights and os.path.exists(saved_model_weights_file_path):
        east_network.load_weights(saved_model_weights_file_path)

    east_network.fit_generator(generator=gen(args.batch_size, args.data_dir),
                               steps_per_epoch=cfg.steps_per_epoch(args.validation_split_ratio, args.batch_size,
                                                                   total_img),
                               epochs=args.epoch_num,
                               validation_data=gen(args.batch_size, args.data_dir, is_val=True),
                               validation_steps=cfg.validation_steps(args.validation_split_ratio, args.batch_size,
                                                                     total_img),
                               verbose=1,
                               initial_epoch=args.initial_epoch,
                               callbacks=[
                                   EarlyStopping(patience=args.patience, verbose=1),
                                   ModelCheckpoint(filepath=model_weights_path,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   verbose=1)])
    east_network.save(saved_model_file_path)
    east_network.save_weights(saved_model_weights_file_path)
