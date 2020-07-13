import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from mobilenet import _MobileNet

import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate age estimation model "
                                                 "using the APPA-REAL validation data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    weight_file = args.weight_file

    # load model and weights
    img_size = 128
    model = _MobileNet(depth=1)()
    model.load_weights(weight_file)
    dataset_root = Path(__file__).parent.joinpath("appa-real")
    validation_image_dir = dataset_root.joinpath("imgs")
    gt_valid_path = dataset_root.joinpath("labels.csv")
    with open(dataset_root.joinpath("ignore_list.txt")) as f:
        ignore_list = []
        for line in f:
            ignore_list.append(line.strip())
    #print(ignore_list)
    df = pd.read_csv(str(gt_valid_path))
    gender = [0 for i in range(len(df))]
    df['gender'] = gender
    ages = []
    image_names = []

    for i, row in tqdm(df.iterrows()):
        image_path = dataset_root.joinpath("imgs", row.file_name)
        if row.file_name in ignore_list:
            print(image_path,"ignored")
            df.drop(i)
            continue
        img = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /=  255.
        img  -=  0.5
        img  *=  2.
        img = np.expand_dims(img, 0)
        results = model.predict(img)
        male = np.argmax(results[0][0])
        if male: df['gender'][i] = 'male'
        if not male: df['gender'][i] = 'female'
        #if i == 10:
        #    print(df)
        #    break
        #faces[i % batch_size] = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        #results = model.predict(faces)
        #image_names.append(image_path.name[:-9])

        #if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
            #results = model.predict(faces)
            #predicted_genders = results[0]
            #ages_out = np.arange(0, 101).reshape(101, 1)
            #predicted_ages = results[1].dot(ages_out).flatten()
            #ages += list(predicted_ages)
            # len(ages) can be larger than len(image_names) due to the last batch, but it's ok.

    #name2age = {image_names[i]: ages[i] for i in range(len(image_names))}

    #appa_abs_error = 0.0
    #real_abs_error = 0.0

    #for i, row in df.iterrows():
    #    appa_abs_error += abs(name2age[row.file_name] - row.apparent_age_avg)
    #    real_abs_error += abs(name2age[row.file_name] - row.real_age)

    #print("MAE Apparent: {}".format(appa_abs_error / len(image_names)))
    #print("MAE Real: {}".format(real_abs_error / len(image_names)))

    df.to_csv(str(gt_valid_path))
if __name__ == '__main__':
    main()
