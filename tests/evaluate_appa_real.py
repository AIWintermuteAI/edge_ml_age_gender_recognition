import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

ages = ['0-10', '11-20', '21-45', '46-60', '60-100']

def age_to_groups(age):

    for group in ages:
        if age >= int(group.split('-')[0]) and age <= int(group.split('-')[1]):
            age = ages.index(group)

    return age

def get_args():
    parser = argparse.ArgumentParser(description="This script evaluate age estimation model "
                                                 "using the APPA-REAL validation data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model_file = args.model_file

    # load model and weights
    img_size = 128
    batch_size = 64
    model = load_model(model_file)

    dataset_root = Path(__file__).parent.joinpath("../data/processed_data")

    test_image_dir = dataset_root.joinpath("test")
    test_csv_path = dataset_root.joinpath("test.csv")

    df = pd.read_csv(str(test_csv_path))

    ages = []
    genders = []

    faces = np.empty((batch_size, img_size, img_size, 3))

    for i, row in tqdm(df.iterrows()):
        image_path = dataset_root.joinpath("test", row.file)

        img = cv2.resize(cv2.imread(str(image_path), 1), (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /=  255.
        img -=  0.5
        img *=  2.
        faces[i % batch_size] = img

        df.at[i, 'age'] = age_to_groups(df.at[i, 'age'])

        if (i + 1) % batch_size == 0 or i == len(df) - 1:
            results = model.predict(faces)
            predicted_genders = results[0]
            predicted_ages = results[1]*100
            genders += list(predicted_genders)
            ages += list(predicted_ages)
            # len(ages) can be larger than len(image_names) due to the last batch, but it's ok.

    appa_abs_error = 0.0
    real_abs_error = 0.0
    accuracy_age = 0.0
    accuracy_gender = 0.0

    for i, row in df.iterrows():

        accuracy_age += 1 if np.argmax(ages[i]) == row.age else 0        
        accuracy_gender += 1 if genders[i].round() == row.gender else 0

    print("Accuracy age: {}".format(accuracy_age / len(df)))
    print("Accuracy gender: {}".format(accuracy_gender / len(df)))


if __name__ == '__main__':
    main()
