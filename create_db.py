import numpy as np
import cv2
import scipy.io
import argparse
import os
from tqdm import tqdm
from utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=32,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    sample_num = len(face_score)
    out_imgs = np.memmap('filename.myarray', dtype='object', mode='w+', shape=(sample_num, 1))
    valid_sample_num = 0

    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        out_imgs[valid_sample_num] = full_path[i]
        valid_sample_num += 1

    output = {"image": out_imgs[:valid_sample_num], "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
