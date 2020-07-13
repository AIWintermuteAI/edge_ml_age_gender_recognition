import pandas as pd
import math
import logging
import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from mobilenet import _MobileNet
from utils import load_data
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
import tensorflow as tf
logging.basicConfig(level=logging.DEBUG)
from datetime import datetime
import os

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--bottleneck_weights", type=str,
                        help="bottleneck_weights")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default=os.path.join("checkpoints", datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                        help="checkpoint dir")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    input_path = args.input
    _format = input_path.split(".")[1]
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    depth = args.depth
    validation_split = args.validation_split
    use_augmentation = args.aug
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(output_path)
    logging.debug("Loading data...")
    image_list, gender, age, _ ,  _ = load_data(input_path, _format)
    print(image_list, gender, age)
    total = age.shape[0]
    class_weights={'pred_gender': {0: 1, 1: 1}, 'pred_age': {0: 5, 1: 5, 2: 1, 3: 1, 4: 3}}
    for i in range(5):
        age_category_count = np.count_nonzero(age == i)
        print(age_category_count)
        class_weights['pred_age'][i] = math.ceil((1 / age_category_count)*(total)/5)


    print(class_weights)
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, 5)
    model = _MobileNet(depth=depth)()
    if args.bottleneck_weights:
        model.load_weights(args.bottleneck_weights,by_name=True)
        print('Transfer learning mode')
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss={'pred_age': 'categorical_crossentropy', 'pred_gender': 'categorical_crossentropy'}, metrics={'pred_age': 'accuracy', 'pred_gender': 'accuracy'})
    #model.compile(optimizer=opt, loss={'pred_age': 'mse', 'pred_gender': 'categorical_crossentropy'}, metrics={'pred_age': 'mae', 'pred_gender': 'accuracy'}, loss_weights={'pred_age': 0.25, 'pred_gender': 10})

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    callbacks = [ReduceLROnPlateau(min_delta=0.000001,monitor="val_pred_age_accuracy",verbose=1, save_best_only=True,mode="auto"),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_pred_age_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")]

    logging.debug("Running training...")

    data_num = len(image_list)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = image_list[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]

    train_num = int(data_num * (1 - validation_split))
    test_num = int(data_num * validation_split)
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    if use_augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.7,1.3],
            horizontal_flip=True, preprocessing_function=get_random_eraser(v_l=0, v_h=255))
        training_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2, datagen=datagen)()
        validation_generator = MixupGenerator(X_test, [y_test_g, y_test_a], batch_size=batch_size, alpha=0.2)()

        hist = model.fit_generator(generator=training_generator,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=test_num // batch_size,
                                   epochs=nb_epochs, verbose=1,
                                   callbacks=callbacks)#, class_weight=class_weights)
    else:
        hist = model.fit(X_train, [y_train_g, y_train_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
                         validation_data=(X_test, [y_test_g, y_test_a]))

    logging.debug("Saving history...")
    pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")


if __name__ == '__main__':
    main()
