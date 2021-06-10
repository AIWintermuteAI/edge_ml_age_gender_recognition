import os
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

ignore_list = ['000025.jpg', '000049.jpg', '000067.jpg', '000085.jpg', '000095.jpg', '000100.jpg', '000127.jpg', '000145.jpg', '000191.jpg', '000215.jpg', '000320.jpg', '000373.jpg', '000392.jpg', '000407.jpg', '000488.jpg', '000503.jpg', '000506.jpg', '000510.jpg', '000536.jpg', '000605.jpg', '000625.jpg', '000639.jpg', '000707.jpg', '000708.jpg', '000712.jpg', '000813.jpg', '000837.jpg', '000848.jpg', '000856.jpg', '000891.jpg', '000892.jpg', '001022.jpg', '001044.jpg', '001095.jpg', '001098.jpg', '001122.jpg', '001125.jpg', '001137.jpg', '001156.jpg', '001227.jpg', '001251.jpg', '001267.jpg', '001282.jpg', '001328.jpg', '001349.jpg', '001380.jpg', '001427.jpg', '001460.jpg', '001475.jpg', '001697.jpg', '001744.jpg', '001864.jpg', '001957.jpg', '001968.jpg', '001973.jpg', '002029.jpg', '002063.jpg', '002109.jpg', '002112.jpg', '002115.jpg', '002123.jpg', '002162.jpg', '002175.jpg', '002179.jpg', '002221.jpg', '002250.jpg', '002303.jpg', '002359.jpg', '002360.jpg', '002412.jpg', '002417.jpg', '002435.jpg', '002460.jpg', '002466.jpg', '002472.jpg', '002488.jpg', '002535.jpg', '002543.jpg', '002565.jpg', '002615.jpg', '002630.jpg', '002633.jpg', '002661.jpg', '002733.jpg', '002756.jpg', '002860.jpg', '002883.jpg', '002887.jpg', '002890.jpg', '002948.jpg', '002995.jpg', '003018.jpg', '003130.jpg', '003164.jpg', '003233.jpg', '003258.jpg', '003271.jpg', '003329.jpg', '003351.jpg', '003357.jpg', '003371.jpg', '003415.jpg', '003427.jpg', '003441.jpg', '003447.jpg', '003458.jpg', '003570.jpg', '003625.jpg', '003669.jpg', '003711.jpg', '003747.jpg', '003749.jpg', '003758.jpg', '003763.jpg', '003772.jpg', '003805.jpg', '003814.jpg', '003903.jpg']

def get_args():
    parser = argparse.ArgumentParser(description="This script creates database for training from the UTKFace dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_size", type=int, default=128,
                        help="output image size")
    args = parser.parse_args()
    return args


def get_appareal(subset, img_size):
    global ignore_list

    root = os.path.join('data', 'appa-real-release')
    anns = 'allcategories_' + subset + '.csv'
    ages = 'gt_avg_' + subset + '.csv'

    appareal_anns = os.path.join(root, anns)
    appareal_ages = os.path.join(root, ages)

    appareal_df = pd.read_csv(appareal_anns)
    appareal_age_df = pd.read_csv(appareal_ages)

    appareal_df['age'] = appareal_age_df['apparent_age_avg'].astype(int)

    filenames = []

    genders = {'male': 0, 'female': 1}
    appareal_df = appareal_df[~appareal_df['file'].isin(ignore_list)]
    appareal_df = appareal_df.reset_index(drop=True)
    appareal_df['gender'] = appareal_df['gender'].replace(genders)

    for i, row in tqdm(appareal_df.iterrows()):
        image_path, age, gender, race = row['file'], row['age'], row['gender'], row['race']
        image_path = image_path.split('.')[0] + '.jpg_face.jpg'

        image_name = str(age) + '_' + str(gender) + '_' + str(race) + '_' + str(i) + '.jpg'
        filenames.append(image_name)
        img = cv2.imread(os.path.join('data', 'appa-real-release', subset, image_path))
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join('data', 'processed_data', subset, image_name), img)

    del appareal_df['happiness'], appareal_df['makeup'], appareal_df['time'], appareal_df['race']

    appareal_df['file'] = pd.Series(filenames)
    return appareal_df

def main():
    try:
        os.mkdir('data/processed_data')
        os.mkdir('data/processed_data/train')
        os.mkdir('data/processed_data/valid') 
        os.mkdir('data/processed_data/test')              
    except:
        pass

    args = get_args()

    utk_dir = Path('data/UTKFace40')
    output_path_train = 'data/processed_data/train.csv'
    output_path_valid = 'data/processed_data/valid.csv'
    output_path_test = 'data/processed_data/test.csv'    
    img_size = args.img_size

    out_genders = []
    out_ages = []
    out_img_names = []

    for i, image_path in enumerate(tqdm(utk_dir.glob("*.jpg"))):
        try:
            image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
            age, gender = image_name.split("_")[:2]
            out_genders.append(int(gender))
            out_ages.append(min(int(age), 100))
            out_img_names.append(image_name)
            img = cv2.imread(str(image_path))
            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(os.path.join('data/processed_data/train', image_name), img)
        except Exception as e:
            print(image_path, e)

    output = {"file": out_img_names, "gender": out_genders, "age": out_ages}
    utk_df = pd.DataFrame.from_dict(output)

    appareal_train_df = get_appareal(subset = 'train', img_size = img_size)

    train_df = pd.concat([utk_df, appareal_train_df])  
    train_df.to_csv(output_path_train, index = False)

    valid_df = get_appareal(subset = 'valid', img_size = img_size)    
    valid_df.to_csv(output_path_valid, index = False)

    test_df = get_appareal(subset = 'test', img_size = img_size)    
    test_df.to_csv(output_path_test, index = False)

if __name__ == '__main__':
    main()
