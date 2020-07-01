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
    parser.add_argument("--transfer", action="store_true",
                        help="db for transfer learning")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    min_score = args.min_score

    root_path = "data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    sample_num = len(face_score)
    out_imgs = np.memmap('filename.myarray', dtype='object', mode='w+', shape=(7685, 1))
    valid_sample_num = 0
    age_counters = [0 for i in range(5)]
    females, males = 0, 0
    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        if age[i] > 0 and age[i] <= 10:
            age[i] = 0
            age_counters[0] += 1
            if age_counters[0] >= 1281: continue

        elif age[i] > 10 and age[i] <= 20:
            age[i] = 1
            age_counters[1] += 1
            if age_counters[1] >= 1281: continue

        elif age[i] > 20 and age[i] <= 45:
            age[i] = 2
            age_counters[2] += 1
            if age_counters[2] >= 1281: continue

        elif age[i] > 45 and age[i] <= 60:
            age[i] = 3
            age_counters[3] += 1
            if age_counters[3] >= 1281: continue

        elif age[i] > 60:
            age[i] = 4
            if gender[i] == 0: females += 1
            if gender[i] == 1: males += 1
            age_counters[4] += 1
            if females >= 1281//2 and gender[i] == 0: continue
            if males >= 1281//2 and gender[i] == 1: continue
            #if age_counters[4] >= 2562: continue

        #if age[i] == "child" or age[i] == 'young_adult':
        #    for i in range(4):
        #       out_genders.append(int(gender[i]))
        #        out_ages.append(age[i])
        #        out_imgs[valid_sample_num] = full_path[i]
        #        valid_sample_num += 1

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        out_imgs[valid_sample_num] = full_path[i]
        valid_sample_num += 1

    print(females, males)
    output = {"image": out_imgs[:valid_sample_num], "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "min_score": min_score}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
